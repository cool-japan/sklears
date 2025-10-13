//! Monitoring Configuration System
//!
//! This module provides comprehensive configuration management for the execution monitoring
//! framework. It includes all configuration structures, validation, builder patterns,
//! and default implementations for different deployment scenarios.
//!
//! # Architecture Overview
//!
//! The monitoring configuration system is built on a modular architecture that separates
//! concerns while providing a unified interface for configuration management:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                       Monitoring Configuration System                           │
//! ├─────────────────────────────────────────────────────────────────────────────────┤
//! │  Config Core   │  Metrics      │  Events       │  Performance  │  Resources    │
//! │  (Foundation)  │  (Collection) │  (Tracking)   │  (Profiling)  │  (Capacity)   │
//! ├─────────────────────────────────────────────────────────────────────────────────┤
//! │  Alerts        │  Data Management              │  Config Builder               │
//! │  (Notifications) │  (Retention, Export, Sampling) │  (Builder Pattern, Validation) │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Core Modules
//!
//! ## Config Core (`config_core`)
//! The foundation module containing the main `MonitoringConfig` structure and core types
//! used throughout the monitoring system. Provides common enums, time ranges, and
//! severity levels used by all other modules.
//!
//! ## Metrics Configuration (`metrics_config`)
//! Comprehensive metrics collection configuration including:
//! - Collection intervals and strategies
//! - Aggregation and storage settings
//! - Custom metrics definitions
//! - Performance-optimized collection modes
//!
//! ## Event Tracking (`event_tracking`)
//! Event collection and processing configuration including:
//! - Event types and filtering
//! - Collection methods (push, pull, file-based)
//! - Enrichment rules and context sources
//! - Sampling strategies for high-volume scenarios
//!
//! ## Performance Monitoring (`performance_config`)
//! Performance analysis and profiling configuration including:
//! - CPU, memory, and I/O profiling
//! - Benchmarking and regression detection
//! - Anomaly detection algorithms
//! - Performance thresholds and alerting
//!
//! ## Resource Monitoring (`resource_monitoring`)
//! System resource monitoring and capacity planning including:
//! - CPU, memory, storage, and network monitoring
//! - Resource thresholds and alerting
//! - Capacity planning and forecasting
//! - Resource optimization recommendations
//!
//! ## Alert Configuration (`alert_configuration`)
//! Comprehensive alerting and notification system including:
//! - Alert rules and conditions
//! - Notification channels (email, Slack, PagerDuty, etc.)
//! - Escalation policies and suppression rules
//! - Alert aggregation and deduplication
//!
//! ## Data Management (`data_management`)
//! Data lifecycle management including:
//! - Retention policies and cleanup strategies
//! - Export formats and destinations
//! - Sampling configurations for volume management
//! - Archive and compression settings
//!
//! ## Configuration Builder (`config_builder`)
//! Builder pattern and validation infrastructure including:
//! - Fluent configuration building interface
//! - Environment-specific configuration profiles
//! - Comprehensive validation and error reporting
//! - Health check configuration
//!
//! # Usage Examples
//!
//! ## Quick Start with Defaults
//! ```rust
//! use sklears_compose::monitoring_config::MonitoringConfig;
//!
//! // Use default configuration
//! let config = MonitoringConfig::default();
//!
//! // Use production-optimized configuration
//! let prod_config = MonitoringConfig::production();
//!
//! // Use development-optimized configuration
//! let dev_config = MonitoringConfig::development();
//! ```
//!
//! ## Builder Pattern Configuration
//! ```rust
//! use sklears_compose::monitoring_config::ConfigBuilder;
//! use std::time::Duration;
//!
//! let config = ConfigBuilder::new()
//!     .enable_metrics(true)
//!     .metrics_collection_interval(Duration::from_secs(30))
//!     .enable_alerts(true)
//!     .alert_channel("slack")
//!     .sampling_rate(0.1)
//!     .retention_period(Duration::from_secs(86400 * 90))
//!     .build()?;
//! ```
//!
//! ## Environment-Specific Configuration
//! ```rust
//! // High-volume production environment
//! let high_volume_config = ConfigBuilder::high_volume()
//!     .with_custom_metric("api_requests")
//!     .cpu_thresholds(70.0, 90.0)
//!     .enable_archiving(true)
//!     .build()?;
//!
//! // Security-focused environment
//! let security_config = ConfigBuilder::security_focused()
//!     .event_buffer_size(50000)
//!     .enable_export(false)  // Reduce data exposure
//!     .build()?;
//!
//! // Compliance-focused environment
//! let compliance_config = ConfigBuilder::compliance_focused()
//!     .retention_period(Duration::from_secs(86400 * 365 * 7))  // 7 years
//!     .sampling_rate(1.0)  // No sampling for compliance
//!     .enable_archiving(true)
//!     .build()?;
//! ```
//!
//! ## Advanced Custom Configuration
//! ```rust
//! use sklears_compose::monitoring_config::{
//!     ConfigBuilder, MetricsConfig, AlertConfig, PerformanceMonitoringConfig
//! };
//!
//! let config = ConfigBuilder::new()
//!     .configure(|config| {
//!         // Direct configuration access for advanced use cases
//!         config.metrics.storage.batch_size = 5000;
//!         config.performance.profiling.sampling_rate = 0.05;
//!         config.alerts.aggregation.window_size = Duration::from_secs(600);
//!     })
//!     .from_environment()  // Load from environment variables
//!     .build()?;
//! ```
//!
//! ## Component-Specific Configuration
//! ```rust
//! // Configure just metrics collection
//! let metrics_config = MetricsConfig::high_performance();
//!
//! // Configure just alerting
//! let alert_config = AlertConfig::production();
//!
//! // Configure just performance monitoring
//! let perf_config = PerformanceMonitoringConfig::development_profiling();
//! ```
//!
//! # Configuration Profiles
//!
//! The system provides several pre-configured profiles optimized for different use cases:
//!
//! ## Production Profile
//! - Comprehensive monitoring enabled
//! - Moderate sampling (10%) for efficiency
//! - Full alerting with escalation policies
//! - 90-day retention with archiving
//! - Performance profiling enabled
//! - Security-focused defaults
//!
//! ## High-Volume Profile
//! - Aggressive sampling (1%) for scalability
//! - Large buffers and batch processing
//! - Real-time aggregation
//! - Minimal profiling overhead
//! - Optimized for throughput
//!
//! ## Development Profile
//! - Detailed logging and profiling
//! - No sampling (full data capture)
//! - Minimal alerting to reduce noise
//! - Short retention periods
//! - Enhanced debugging features
//!
//! ## Security-Focused Profile
//! - Comprehensive event tracking
//! - No data export by default
//! - Extended retention for audit trails
//! - Security-specific alert rules
//! - Encryption enabled for archived data
//!
//! ## Compliance-Focused Profile
//! - No sampling (complete data retention)
//! - Extended retention periods (7+ years)
//! - Comprehensive archiving with encryption
//! - Audit-trail optimized event tracking
//! - Tamper-evident data storage
//!
//! # Validation and Error Handling
//!
//! The configuration system provides comprehensive validation:
//!
//! ```rust
//! use sklears_compose::monitoring_config::ConfigBuilder;
//!
//! let result = ConfigBuilder::new()
//!     .sampling_rate(1.5)  // Invalid: > 1.0
//!     .cpu_thresholds(90.0, 80.0)  // Invalid: warning > critical
//!     .build();
//!
//! match result {
//!     Ok(config) => {
//!         // Configuration is valid
//!     },
//!     Err(e) => {
//!         // Handle validation errors
//!         eprintln!("Configuration validation failed: {}", e);
//!     }
//! }
//! ```
//!
//! # Integration with Monitoring System
//!
//! The configuration system integrates seamlessly with the monitoring framework:
//!
//! ```rust
//! use sklears_compose::monitoring_config::ConfigBuilder;
//! use sklears_compose::monitoring::MonitoringFramework;
//!
//! // Build configuration
//! let config = ConfigBuilder::production()
//!     .with_custom_metric("business_kpi")
//!     .alert_threshold("high_cpu", "cpu_usage", 85.0)
//!     .build()?;
//!
//! // Initialize monitoring framework with configuration
//! let framework = MonitoringFramework::new(config)?;
//! framework.start()?;
//! ```
//!
//! # Best Practices
//!
//! ## Configuration Management
//! 1. **Use environment profiles**: Start with predefined profiles and customize as needed
//! 2. **Validate early**: Use the builder pattern to catch configuration errors early
//! 3. **Document customizations**: Keep track of deviations from standard profiles
//! 4. **Test configurations**: Validate configurations in non-production environments first
//!
//! ## Performance Optimization
//! 1. **Adjust sampling rates**: Use appropriate sampling for your data volume
//! 2. **Tune buffer sizes**: Balance memory usage with processing efficiency
//! 3. **Configure aggregation**: Use real-time aggregation for high-volume scenarios
//! 4. **Optimize storage**: Choose appropriate storage backends for your use case
//!
//! ## Security Considerations
//! 1. **Protect sensitive data**: Be careful with data export in sensitive environments
//! 2. **Use encryption**: Enable encryption for archived data in production
//! 3. **Audit configurations**: Track configuration changes for compliance
//! 4. **Principle of least privilege**: Only enable features you actually need

// Core module declarations
pub mod config_core;
pub mod metrics_config;
pub mod event_tracking;
pub mod performance_config;
pub mod resource_monitoring;
pub mod alert_configuration;
pub mod data_management;
pub mod config_builder;

// Re-export core types and configurations for convenience
pub use config_core::{
    MonitoringConfig, TimeRange, SeverityLevel, ComparisonOperator
};

pub use metrics_config::{
    MetricsConfig, MetricType, MetricsAggregationConfig, MetricsStorageConfig,
    AggregationFunction, StorageBackend, FileFormat, CustomMetricDefinition,
    CustomMetricType
};

pub use event_tracking::{
    EventTrackingConfig, EventType, EventCollectionConfig, EventEnrichmentConfig,
    EventFilteringConfig, EventSamplingConfig, CollectionMethod, EnrichmentRule,
    EnrichmentSource, FilterRule, FilterAction
};

pub use performance_config::{
    PerformanceMonitoringConfig, PerformanceMetricType, ProfilingConfig, ProfilingMode,
    ProfilerOutputConfig, ProfilerOutputFormat, BenchmarkingConfig, BenchmarkSuite,
    BenchmarkTest, AnomalyDetectionConfig, AnomalyDetectionAlgorithm, PerformanceThresholds,
    ThresholdConfig, OptimizationAnalysisConfig
};

pub use resource_monitoring::{
    ResourceMonitoringConfig, ResourceType, ResourceThresholds, CpuThresholds,
    MemoryThresholds, StorageThresholds, NetworkThresholds, GpuThresholds,
    CapacityPlanningConfig, ForecastingAlgorithm, ResourceOptimizationConfig,
    ResourceCollectionConfig, ResourceRetentionConfig
};

pub use alert_configuration::{
    AlertConfig, AlertRule, AlertCondition, AlertChannel, AlertAggregationConfig,
    AlertSuppressionConfig, EscalationConfig, AlertManagementConfig, SmtpConfig,
    EmailTemplate, SlackFormatting, PagerDutyService, WebhookAuth
};

pub use data_management::{
    DataRetentionConfig, ExportConfig, SamplingConfig, RetentionPolicy, CleanupConfig,
    ArchiveConfig, ExportFormat, ExportDestination, ExportSchedulingConfig,
    ExportFilteringConfig, SamplingStrategy, AdaptiveSamplingConfig
};

pub use config_builder::{
    ConfigBuilder, EnvironmentProfile, ConfigurationSummary, HealthCheckConfig,
    HealthCheckEndpoint, HealthCriteria, HealthCheckType, HealthCheckAuth
};

// Type aliases for common usage patterns
pub type MonitoringResult<T> = Result<T, MonitoringError>;

/// Common error types for monitoring configuration
#[derive(Debug, thiserror::Error)]
pub enum MonitoringError {
    #[error("Configuration validation error: {0}")]
    ValidationError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Missing required configuration: {0}")]
    MissingConfiguration(String),

    #[error("Configuration conflict: {0}")]
    ConfigurationConflict(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Core error: {0}")]
    CoreError(#[from] sklears_core::error::SklearsError),
}

// Utility functions for common configuration operations
/// Create a basic monitoring configuration with sensible defaults
pub fn create_basic_config() -> MonitoringConfig {
    MonitoringConfig::default()
}

/// Create a production-ready monitoring configuration
pub fn create_production_config() -> MonitoringConfig {
    MonitoringConfig::production()
}

/// Create a development-friendly monitoring configuration
pub fn create_development_config() -> MonitoringConfig {
    MonitoringConfig::development()
}

/// Create a configuration builder with environment-specific defaults
pub fn builder_for_environment(env: &str) -> ConfigBuilder {
    match env.to_lowercase().as_str() {
        "production" | "prod" => ConfigBuilder::production(),
        "development" | "dev" => ConfigBuilder::development(),
        "testing" | "test" => ConfigBuilder::testing(),
        "high-volume" | "high_volume" => ConfigBuilder::high_volume(),
        "security" | "security-focused" => ConfigBuilder::security_focused(),
        "compliance" | "compliance-focused" => ConfigBuilder::compliance_focused(),
        _ => ConfigBuilder::new(),
    }
}

/// Validate a monitoring configuration and return detailed error information
pub fn validate_config(config: &MonitoringConfig) -> MonitoringResult<Vec<String>> {
    let mut warnings = Vec::new();

    // Perform comprehensive validation
    if let Err(e) = config.validate() {
        return Err(MonitoringError::ValidationError(e.to_string()));
    }

    // Check for potential performance issues
    if config.metrics.enabled && config.metrics.collection_interval < Duration::from_secs(1) {
        warnings.push("Very short metrics collection interval may impact performance".to_string());
    }

    if config.sampling.enabled && config.sampling.rate > 0.5 {
        warnings.push("High sampling rate may not provide significant volume reduction".to_string());
    }

    if config.events.enabled && config.events.collection.buffer_size > 100000 {
        warnings.push("Large event buffer size may consume significant memory".to_string());
    }

    // Check for security considerations
    if config.export.enabled && !config.retention.archive.encryption.enabled {
        warnings.push("Data export is enabled but encryption is disabled".to_string());
    }

    // Check for operational considerations
    if config.alerts.enabled && config.alerts.channels.is_empty() {
        warnings.push("Alerting is enabled but no notification channels are configured".to_string());
    }

    if !config.health_checks.enabled {
        warnings.push("Health checks are disabled - consider enabling for better observability".to_string());
    }

    Ok(warnings)
}

/// Load monitoring configuration from environment variables with fallback to defaults
pub fn load_config_from_env() -> MonitoringResult<MonitoringConfig> {
    let env_profile = std::env::var("MONITORING_PROFILE")
        .unwrap_or_else(|_| "development".to_string());

    let config = builder_for_environment(&env_profile)
        .from_environment()
        .build()
        .map_err(|e| MonitoringError::ValidationError(e.to_string()))?;

    Ok(config)
}

/// Export monitoring configuration to JSON for storage or sharing
pub fn export_config_to_json(config: &MonitoringConfig) -> MonitoringResult<String> {
    serde_json::to_string_pretty(config)
        .map_err(MonitoringError::SerializationError)
}

/// Import monitoring configuration from JSON
pub fn import_config_from_json(json_str: &str) -> MonitoringResult<MonitoringConfig> {
    let config: MonitoringConfig = serde_json::from_str(json_str)
        .map_err(MonitoringError::SerializationError)?;

    // Validate imported configuration
    config.validate()
        .map_err(|e| MonitoringError::ValidationError(e.to_string()))?;

    Ok(config)
}

/// Compare two monitoring configurations and return differences
pub fn compare_configs(config1: &MonitoringConfig, config2: &MonitoringConfig) -> Vec<String> {
    let mut differences = Vec::new();

    // Compare major settings
    if config1.metrics.enabled != config2.metrics.enabled {
        differences.push(format!(
            "Metrics enabled: {} vs {}",
            config1.metrics.enabled, config2.metrics.enabled
        ));
    }

    if config1.events.enabled != config2.events.enabled {
        differences.push(format!(
            "Events enabled: {} vs {}",
            config1.events.enabled, config2.events.enabled
        ));
    }

    if config1.alerts.enabled != config2.alerts.enabled {
        differences.push(format!(
            "Alerts enabled: {} vs {}",
            config1.alerts.enabled, config2.alerts.enabled
        ));
    }

    if config1.sampling.rate != config2.sampling.rate {
        differences.push(format!(
            "Sampling rate: {} vs {}",
            config1.sampling.rate, config2.sampling.rate
        ));
    }

    if config1.retention.metrics_retention != config2.retention.metrics_retention {
        differences.push(format!(
            "Metrics retention: {:?} vs {:?}",
            config1.retention.metrics_retention, config2.retention.metrics_retention
        ));
    }

    differences
}

/// Get configuration recommendations based on deployment scenario
pub fn get_config_recommendations(scenario: &str) -> Vec<String> {
    let mut recommendations = Vec::new();

    match scenario.to_lowercase().as_str() {
        "startup" => {
            recommendations.push("Start with development profile for cost efficiency".to_string());
            recommendations.push("Enable basic metrics and health checks".to_string());
            recommendations.push("Use minimal alerting to avoid noise".to_string());
            recommendations.push("Short retention periods to minimize storage costs".to_string());
        },
        "growth" => {
            recommendations.push("Upgrade to production profile for reliability".to_string());
            recommendations.push("Enable comprehensive alerting and escalation".to_string());
            recommendations.push("Implement sampling to manage data volume".to_string());
            recommendations.push("Enable performance monitoring for optimization".to_string());
        },
        "enterprise" => {
            recommendations.push("Use high-volume profile for scalability".to_string());
            recommendations.push("Enable archiving for compliance requirements".to_string());
            recommendations.push("Implement security-focused event tracking".to_string());
            recommendations.push("Use advanced capacity planning features".to_string());
        },
        "regulated" => {
            recommendations.push("Use compliance-focused profile".to_string());
            recommendations.push("Disable sampling to ensure complete data retention".to_string());
            recommendations.push("Enable encryption for all archived data".to_string());
            recommendations.push("Implement comprehensive audit trails".to_string());
        },
        _ => {
            recommendations.push("Consider your specific requirements for profile selection".to_string());
            recommendations.push("Start with conservative settings and adjust based on needs".to_string());
            recommendations.push("Enable validation to catch configuration issues early".to_string());
        }
    }

    recommendations
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_basic_config_creation() {
        let config = create_basic_config();
        assert!(config.metrics.enabled);
        assert!(config.health_checks.enabled);
    }

    #[test]
    fn test_production_config_creation() {
        let config = create_production_config();
        assert!(config.metrics.enabled);
        assert!(config.alerts.enabled);
        assert!(config.performance.enabled);
        assert!(config.export.enabled);
    }

    #[test]
    fn test_development_config_creation() {
        let config = create_development_config();
        assert!(config.metrics.enabled);
        assert!(!config.alerts.enabled);
        assert!(!config.export.enabled);
    }

    #[test]
    fn test_builder_for_environment() {
        let prod_builder = builder_for_environment("production");
        let dev_builder = builder_for_environment("development");
        let unknown_builder = builder_for_environment("unknown");

        // Should create appropriate builders
        let prod_config = prod_builder.build().unwrap();
        let dev_config = dev_builder.build().unwrap();
        let unknown_config = unknown_builder.build().unwrap();

        assert!(prod_config.alerts.enabled);
        assert!(!dev_config.alerts.enabled);
        assert!(!unknown_config.alerts.enabled); // Default behavior
    }

    #[test]
    fn test_config_validation() {
        let valid_config = create_production_config();
        let warnings = validate_config(&valid_config).unwrap();
        // Should not have validation errors, may have warnings
    }

    #[test]
    fn test_config_comparison() {
        let config1 = create_production_config();
        let config2 = create_development_config();

        let differences = compare_configs(&config1, &config2);
        assert!(!differences.is_empty()); // Should find differences
        assert!(differences.iter().any(|d| d.contains("Alerts enabled")));
    }

    #[test]
    fn test_json_export_import() {
        let original_config = create_production_config();

        let json = export_config_to_json(&original_config).unwrap();
        let imported_config = import_config_from_json(&json).unwrap();

        // Should be equivalent (though exact equality might not work due to serialization)
        assert_eq!(original_config.metrics.enabled, imported_config.metrics.enabled);
        assert_eq!(original_config.alerts.enabled, imported_config.alerts.enabled);
    }

    #[test]
    fn test_recommendations() {
        let startup_recs = get_config_recommendations("startup");
        let enterprise_recs = get_config_recommendations("enterprise");

        assert!(!startup_recs.is_empty());
        assert!(!enterprise_recs.is_empty());
        assert_ne!(startup_recs, enterprise_recs); // Should be different
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        let config = ConfigBuilder::new()
            .enable_metrics(true)
            .enable_events(true)
            .enable_performance_monitoring(true)
            .enable_alerts(true)
            .sampling_rate(0.1)
            .build()
            .unwrap();

        assert!(config.metrics.enabled);
        assert!(config.events.enabled);
        assert!(config.performance.enabled);
        assert!(config.alerts.enabled);
        assert_eq!(config.sampling.rate, 0.1);
    }
}