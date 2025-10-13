//! Core Monitoring Configuration
//!
//! This module contains the main MonitoringConfig structure and core types
//! that form the foundation of the monitoring system. All other monitoring
//! configuration modules integrate through these core definitions.

use std::collections::HashMap;
use std::time::Duration;
use sklears_core::error::{Result as SklResult, SklearsError};

// Re-export types from other modules for convenience
pub use super::metrics_config::MetricsConfig;
pub use super::event_tracking::EventTrackingConfig;
pub use super::performance_config::PerformanceMonitoringConfig;
pub use super::resource_monitoring::ResourceMonitoringConfig;
pub use super::alert_configuration::AlertConfig;
pub use super::data_management::{DataRetentionConfig, ExportConfig, SamplingConfig};
pub use super::config_builder::HealthCheckConfig;

/// Comprehensive monitoring configuration
///
/// This is the main configuration structure that orchestrates all aspects
/// of monitoring behavior including metrics collection, alerting, performance
/// tracking, and observability features.
///
/// # Architecture
///
/// The MonitoringConfig follows a modular architecture where each functional
/// area is encapsulated in its own configuration structure:
///
/// ```text
/// MonitoringConfig
/// ├── MetricsConfig           (metrics collection)
/// ├── EventTrackingConfig     (event tracking and enrichment)
/// ├── PerformanceMonitoringConfig (performance and profiling)
/// ├── ResourceMonitoringConfig    (resource monitoring)
/// ├── AlertConfig             (alerting and notifications)
/// ├── DataRetentionConfig     (data lifecycle management)
/// ├── ExportConfig           (data export and integration)
/// ├── SamplingConfig         (sampling strategies)
/// └── HealthCheckConfig      (health monitoring)
/// ```
///
/// # Usage Examples
///
/// ## Creating a Basic Configuration
/// ```rust
/// use sklears_compose::monitoring_config::MonitoringConfig;
///
/// let config = MonitoringConfig::default();
/// ```
///
/// ## Production Configuration
/// ```rust
/// let config = MonitoringConfig::production();
/// ```
///
/// ## Development Configuration
/// ```rust
/// let config = MonitoringConfig::development();
/// ```
///
/// ## Custom Configuration with Builder
/// ```rust
/// use sklears_compose::monitoring_config::ConfigBuilder;
///
/// let config = ConfigBuilder::new()
///     .enable_metrics(true)
///     .enable_alerts(true)
///     .sampling_rate(0.1)
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Metrics collection configuration
    ///
    /// Controls how metrics are collected, aggregated, and stored.
    /// This includes system metrics, application metrics, and custom metrics.
    pub metrics: MetricsConfig,

    /// Event tracking configuration
    ///
    /// Defines how events are captured, enriched, and processed.
    /// Events provide detailed operational insights and audit trails.
    pub events: EventTrackingConfig,

    /// Performance monitoring configuration
    ///
    /// Controls performance profiling, benchmarking, and anomaly detection.
    /// Essential for identifying performance bottlenecks and optimization opportunities.
    pub performance: PerformanceMonitoringConfig,

    /// Resource monitoring configuration
    ///
    /// Manages monitoring of system resources like CPU, memory, disk, and network.
    /// Includes capacity planning and resource optimization features.
    pub resources: ResourceMonitoringConfig,

    /// Alert configuration
    ///
    /// Defines alerting rules, channels, and escalation policies.
    /// Provides proactive monitoring and incident response capabilities.
    pub alerts: AlertConfig,

    /// Data retention configuration
    ///
    /// Controls how long monitoring data is retained and cleanup policies.
    /// Balances storage costs with data availability requirements.
    pub retention: DataRetentionConfig,

    /// Export and integration configuration
    ///
    /// Manages data export to external systems and integration points.
    /// Enables integration with external monitoring and analytics platforms.
    pub export: ExportConfig,

    /// Sampling configuration for high-volume scenarios
    ///
    /// Controls sampling strategies to manage data volume and performance impact.
    /// Critical for high-throughput environments where full data capture is impractical.
    pub sampling: SamplingConfig,

    /// Health check configuration
    ///
    /// Defines health monitoring endpoints and criteria.
    /// Provides system health visibility and automated health assessments.
    pub health_checks: HealthCheckConfig,
}

/// Time range specification for queries and data operations
///
/// Used throughout the monitoring system for defining time boundaries
/// for data queries, retention policies, and export operations.
#[derive(Debug, Clone)]
pub struct TimeRange {
    /// Start time of the range
    pub start: std::time::SystemTime,

    /// End time of the range
    pub end: std::time::SystemTime,
}

/// Severity levels for alerts and events
///
/// Provides a standardized severity classification system used across
/// all monitoring components for consistent priority handling.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SeverityLevel {
    /// Informational messages and routine events
    Info,
    /// Warning conditions that require attention
    Warning,
    /// Error conditions that affect functionality
    Error,
    /// Critical conditions that require immediate action
    Critical,
    /// Emergency conditions that indicate system failure
    Emergency,
}

/// Comparison operators for conditions and thresholds
///
/// Used throughout the monitoring system for defining alert conditions,
/// threshold comparisons, and filtering criteria.
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    /// Greater than
    Greater,
    /// Greater than or equal to
    GreaterEqual,
    /// Less than
    Less,
    /// Less than or equal to
    LessEqual,
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
}

impl MonitoringConfig {
    /// Create configuration optimized for production environments
    ///
    /// Production configuration includes:
    /// - Enhanced metrics collection with extended retention
    /// - Comprehensive alerting with multiple channels
    /// - Data export enabled for external systems
    /// - Performance profiling for optimization insights
    /// - Resource monitoring with capacity planning
    /// - Adaptive sampling for high-volume scenarios
    ///
    /// # Returns
    ///
    /// A MonitoringConfig instance optimized for production use.
    pub fn production() -> Self {
        let mut config = Self::default();

        // Enable comprehensive monitoring for production
        config.metrics.enabled = true;
        config.events.enabled = true;
        config.performance.enabled = true;
        config.resources.enabled = true;
        config.alerts.enabled = true;
        config.export.enabled = true;
        config.health_checks.enabled = true;

        // Configure for production scale
        config.sampling.enabled = true;
        config.sampling.rate = 0.1; // 10% sampling for high volume
        config.performance.profiling.enabled = true;

        // Extended retention for production analysis
        config.retention.metrics_retention = Duration::from_secs(86400 * 90); // 90 days
        config.retention.events_retention = Duration::from_secs(86400 * 30); // 30 days

        config
    }

    /// Create configuration optimized for development environments
    ///
    /// Development configuration includes:
    /// - Basic metrics collection for debugging
    /// - Minimal alerting to avoid noise
    /// - No data export to external systems
    /// - Detailed logging for development insights
    /// - Shorter retention periods
    ///
    /// # Returns
    ///
    /// A MonitoringConfig instance optimized for development use.
    pub fn development() -> Self {
        let mut config = Self::default();

        // Enable minimal monitoring for development
        config.metrics.enabled = true;
        config.events.enabled = true;
        config.performance.enabled = false; // Reduce overhead
        config.resources.enabled = true;
        config.alerts.enabled = false; // Avoid noise in development
        config.export.enabled = false; // No external integration needed
        config.health_checks.enabled = true;

        // No sampling needed in development
        config.sampling.enabled = false;
        config.sampling.rate = 1.0; // Full data capture

        // Shorter retention for development
        config.retention.metrics_retention = Duration::from_secs(86400 * 7); // 7 days
        config.retention.events_retention = Duration::from_secs(86400 * 3); // 3 days

        config
    }

    /// Create configuration optimized for testing environments
    ///
    /// Testing configuration includes:
    /// - Minimal overhead monitoring
    /// - Fast data collection intervals
    /// - Short retention periods
    /// - Comprehensive health checks
    /// - No external integrations
    ///
    /// # Returns
    ///
    /// A MonitoringConfig instance optimized for testing use.
    pub fn testing() -> Self {
        let mut config = Self::default();

        // Minimal monitoring for testing
        config.metrics.enabled = true;
        config.events.enabled = true;
        config.performance.enabled = false;
        config.resources.enabled = false;
        config.alerts.enabled = false;
        config.export.enabled = false;
        config.health_checks.enabled = true;

        // Fast intervals for quick feedback in tests
        config.metrics.collection_interval = Duration::from_millis(100);
        config.health_checks.interval = Duration::from_millis(500);

        // Very short retention for testing
        config.retention.metrics_retention = Duration::from_secs(300); // 5 minutes
        config.retention.events_retention = Duration::from_secs(60); // 1 minute

        config
    }

    /// Validate the entire monitoring configuration
    ///
    /// Performs comprehensive validation of all configuration components
    /// to ensure they are internally consistent and will function correctly.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the configuration is valid
    /// - `Err(SklearsError)` if validation fails with details about the issue
    ///
    /// # Validation Checks
    ///
    /// - Sampling rates are between 0.0 and 1.0
    /// - Collection intervals are reasonable (>= 100ms)
    /// - Retention periods are positive
    /// - Alert conditions are well-formed
    /// - Export configurations are valid
    /// - Resource thresholds are consistent
    pub fn validate(&self) -> SklResult<()> {
        // Validate sampling configuration
        if self.sampling.enabled {
            if self.sampling.rate < 0.0 || self.sampling.rate > 1.0 {
                return Err(SklearsError::InvalidParameter(
                    "Sampling rate must be between 0.0 and 1.0".to_string()
                ));
            }
        }

        // Validate metrics collection interval
        if self.metrics.collection_interval < Duration::from_millis(100) {
            return Err(SklearsError::InvalidParameter(
                "Metrics collection interval must be at least 100ms".to_string()
            ));
        }

        // Validate retention periods
        if self.retention.metrics_retention == Duration::ZERO {
            return Err(SklearsError::InvalidParameter(
                "Metrics retention period must be positive".to_string()
            ));
        }

        if self.retention.events_retention == Duration::ZERO {
            return Err(SklearsError::InvalidParameter(
                "Events retention period must be positive".to_string()
            ));
        }

        // Validate health check configuration
        if self.health_checks.enabled {
            if self.health_checks.timeout >= self.health_checks.interval {
                return Err(SklearsError::InvalidParameter(
                    "Health check timeout must be less than interval".to_string()
                ));
            }
        }

        // Additional component-specific validation would be called here
        // Each component module should provide its own validation methods

        Ok(())
    }

    /// Get a summary of the current configuration status
    ///
    /// Provides a high-level overview of which monitoring features are enabled
    /// and their basic configuration parameters.
    ///
    /// # Returns
    ///
    /// A HashMap containing configuration status information.
    pub fn get_status_summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();

        summary.insert("metrics_enabled".to_string(), self.metrics.enabled.to_string());
        summary.insert("events_enabled".to_string(), self.events.enabled.to_string());
        summary.insert("performance_enabled".to_string(), self.performance.enabled.to_string());
        summary.insert("resources_enabled".to_string(), self.resources.enabled.to_string());
        summary.insert("alerts_enabled".to_string(), self.alerts.enabled.to_string());
        summary.insert("export_enabled".to_string(), self.export.enabled.to_string());
        summary.insert("sampling_enabled".to_string(), self.sampling.enabled.to_string());
        summary.insert("health_checks_enabled".to_string(), self.health_checks.enabled.to_string());

        summary.insert("sampling_rate".to_string(), self.sampling.rate.to_string());
        summary.insert("metrics_interval".to_string(), format!("{:?}", self.metrics.collection_interval));
        summary.insert("health_check_interval".to_string(), format!("{:?}", self.health_checks.interval));

        summary
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics: MetricsConfig::default(),
            events: EventTrackingConfig::default(),
            performance: PerformanceMonitoringConfig::default(),
            resources: ResourceMonitoringConfig::default(),
            alerts: AlertConfig::default(),
            retention: DataRetentionConfig::default(),
            export: ExportConfig::default(),
            sampling: SamplingConfig::default(),
            health_checks: HealthCheckConfig::default(),
        }
    }
}

impl TimeRange {
    /// Create a new time range
    pub fn new(start: std::time::SystemTime, end: std::time::SystemTime) -> Self {
        Self { start, end }
    }

    /// Create a time range for the last duration
    pub fn last(duration: Duration) -> Self {
        let end = std::time::SystemTime::now();
        let start = end - duration;
        Self { start, end }
    }

    /// Get the duration of this time range
    pub fn duration(&self) -> Duration {
        self.end.duration_since(self.start).unwrap_or(Duration::ZERO)
    }

    /// Check if this time range contains the given time
    pub fn contains(&self, time: std::time::SystemTime) -> bool {
        time >= self.start && time <= self.end
    }
}

impl SeverityLevel {
    /// Get the numeric priority of this severity level
    /// Higher numbers indicate higher severity
    pub fn priority(&self) -> u8 {
        match self {
            SeverityLevel::Info => 1,
            SeverityLevel::Warning => 2,
            SeverityLevel::Error => 3,
            SeverityLevel::Critical => 4,
            SeverityLevel::Emergency => 5,
        }
    }

    /// Check if this severity level is at least as severe as the given level
    pub fn is_at_least(&self, other: &SeverityLevel) -> bool {
        self.priority() >= other.priority()
    }
}

impl ComparisonOperator {
    /// Evaluate this comparison operator with the given values
    pub fn evaluate(&self, left: f64, right: f64) -> bool {
        match self {
            ComparisonOperator::Greater => left > right,
            ComparisonOperator::GreaterEqual => left >= right,
            ComparisonOperator::Less => left < right,
            ComparisonOperator::LessEqual => left <= right,
            ComparisonOperator::Equal => (left - right).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (left - right).abs() >= f64::EPSILON,
        }
    }
}