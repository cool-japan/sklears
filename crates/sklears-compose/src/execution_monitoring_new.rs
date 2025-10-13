//! Advanced Execution Monitoring and Observability Framework
//!
//! This module provides a comprehensive execution monitoring and observability system for
//! high-performance machine learning workloads. The framework is designed with modularity,
//! scalability, and enterprise-grade reliability in mind.
//!
//! ## Architecture Overview
//!
//! The execution monitoring framework is organized into specialized modules that work together
//! to provide comprehensive observability:
//!
//! - **Core Monitoring**: Fundamental traits, interfaces, and type definitions
//! - **Session Management**: Lifecycle management and metadata tracking for monitoring sessions
//! - **Metrics Collection**: Performance metrics gathering, storage, and aggregation
//! - **Event Tracking**: Task execution events, filtering, and processing
//! - **Alerting System**: Real-time alert management, channels, and suppression
//! - **Health Monitoring**: System and component health assessment
//! - **Performance Analysis**: Advanced analytics, anomaly detection, and insights
//! - **Configuration**: Comprehensive configuration management and validation
//! - **Reporting**: Report generation, formatting, and visualization
//! - **Data Retention**: Data lifecycle management, archiving, and export
//!
//! ## Key Features
//!
//! ### Real-Time Monitoring
//! - **Live Metrics**: Real-time performance metrics collection and visualization
//! - **Event Streaming**: Continuous event processing with buffering and batching
//! - **Health Checking**: Proactive system health monitoring with automated checks
//! - **Resource Tracking**: CPU, memory, I/O, and custom resource utilization monitoring
//!
//! ### Advanced Analytics
//! - **Anomaly Detection**: Machine learning-based anomaly detection algorithms
//! - **Performance Insights**: Automated performance analysis and optimization recommendations
//! - **Trend Analysis**: Historical trend identification and forecasting
//! - **Correlation Discovery**: Automatic correlation analysis between metrics and events
//!
//! ### Enterprise Features
//! - **Multi-Session Support**: Concurrent monitoring of multiple execution sessions
//! - **Configurable Retention**: Flexible data retention policies with archiving
//! - **Alert Management**: Sophisticated alerting with suppression and escalation
//! - **Export Capabilities**: Multiple export formats (JSON, CSV, Parquet, ProtoBuf)
//!
//! ### High Performance
//! - **Efficient Storage**: Optimized in-memory and persistent storage backends
//! - **Batch Processing**: High-throughput batch processing with configurable batching
//! - **Resource Optimization**: Minimal overhead monitoring with adaptive sampling
//! - **Parallel Processing**: Multi-threaded processing for large-scale deployments
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use sklears_compose::execution_monitoring::*;
//!
//! // Create monitoring configuration
//! let config = MonitoringConfig::builder()
//!     .enable_metrics_collection(true)
//!     .enable_health_checks(true)
//!     .enable_anomaly_detection(true)
//!     .build();
//!
//! // Initialize monitoring system
//! let mut monitor = DefaultExecutionMonitor::new(config.clone());
//!
//! // Start monitoring session
//! let session = monitor.start_monitoring("my_session".to_string(), config)?;
//!
//! // Record performance metrics
//! let metric = PerformanceMetric::new()
//!     .name("cpu_usage")
//!     .value(0.75)
//!     .unit("percentage")
//!     .build();
//!
//! monitor.record_performance_metric(session.session_id.clone(), metric)?;
//!
//! // Get real-time status
//! let status = monitor.get_monitoring_status(&session.session_id)?;
//! println!("Session status: {:?}", status);
//!
//! // Generate comprehensive report
//! let report_config = ReportConfig::default();
//! let report = monitor.generate_report(&session.session_id, report_config)?;
//! ```
//!
//! ## Configuration Examples
//!
//! ### Basic Monitoring
//! ```rust,ignore
//! let config = MonitoringConfig {
//!     metrics_collection: MetricsCollectionConfig {
//!         enabled: true,
//!         collection_interval: Duration::from_secs(1),
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! ### Advanced Alerting
//! ```rust,ignore
//! let alert_config = AlertConfig {
//!     enabled: true,
//!     rules: vec![
//!         AlertRule {
//!             name: "High CPU Usage".to_string(),
//!             condition: AlertCondition::Threshold {
//!                 metric: "cpu_usage".to_string(),
//!                 operator: ComparisonOperator::Greater,
//!                 value: 0.9,
//!                 duration: Duration::from_secs(60),
//!             },
//!             severity: SeverityLevel::Critical,
//!             channels: vec![AlertChannel::Email {
//!                 recipients: vec!["admin@company.com".to_string()]
//!             }],
//!         }
//!     ],
//!     ..Default::default()
//! };
//! ```
//!
//! ### Anomaly Detection
//! ```rust,ignore
//! let anomaly_config = AnomalyDetectionConfig {
//!     enabled: true,
//!     algorithms: vec![
//!         AnomalyDetectionAlgorithm::StatisticalOutlier,
//!         AnomalyDetectionAlgorithm::IsolationForest,
//!     ],
//!     parameters: AnomalyDetectionParameters {
//!         sensitivity: 0.95,
//!         window_size: Duration::from_secs(300),
//!         min_score: 0.8,
//!         persistence_threshold: Duration::from_secs(60),
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! ## Module Organization
//!
//! This framework follows a clean modular architecture:
//!
//! ```text
//! execution_monitoring/
//! ├── monitoring_core       # Core traits and interfaces
//! ├── session_management    # Session lifecycle and metadata
//! ├── metrics_collection    # Performance metrics and storage
//! ├── event_tracking        # Event processing and filtering
//! ├── alerting_system       # Alert management and channels
//! ├── health_monitoring     # System and component health
//! ├── performance_analysis  # Analytics and anomaly detection
//! ├── configuration         # Configuration management
//! ├── reporting_system      # Report generation and formatting
//! └── data_retention        # Data lifecycle and archiving
//! ```
//!
//! ## Performance Considerations
//!
//! The monitoring framework is designed for minimal performance impact:
//!
//! - **Adaptive Sampling**: Automatic sampling rate adjustment based on system load
//! - **Asynchronous Processing**: Non-blocking metric collection and event processing
//! - **Memory Efficiency**: Configurable buffer sizes and memory limits
//! - **Batch Operations**: Batched metric storage and event processing for throughput
//! - **Resource Limits**: Built-in resource usage limits and monitoring overhead control
//!
//! ## Extensibility
//!
//! The framework supports custom extensions:
//!
//! - **Custom Metrics**: Define domain-specific metrics with custom collection logic
//! - **Custom Alerts**: Implement custom alert conditions and response actions
//! - **Custom Storage**: Pluggable storage backends for metrics and events
//! - **Custom Analysis**: Add custom performance analysis algorithms
//! - **Custom Exporters**: Implement custom data export formats and destinations

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, SystemTime, Instant};
use std::thread;
use std::fmt;

// Import specialized monitoring modules
mod monitoring_core;
mod session_management;
mod metrics_collection;
mod event_tracking;
mod alerting_system;
mod health_monitoring;
mod performance_analysis;
mod configuration_management;
mod reporting_system;
mod data_retention;

// Re-export core monitoring interfaces and types
pub use monitoring_core::*;

// Re-export session management functionality
pub use session_management::{
    MonitoringSession,
    MonitoringSessionStatus,
    SessionMetadata,
    MonitoringSessionState,
    SessionManager,
};

// Re-export metrics collection capabilities
pub use metrics_collection::{
    PerformanceMetric,
    MetricType,
    MetricsStorage,
    InMemoryMetricsStorage,
    MetricsCollector,
    MetricsAggregator,
    MetricsAggregationConfig,
    AggregationType,
};

// Re-export event tracking functionality
pub use event_tracking::{
    TaskExecutionEvent,
    TaskEventType,
    TaskEventDetails,
    EventBuffer,
    EventProcessor,
    EventFilter,
    EventEnricher,
};

// Re-export alerting system
pub use alerting_system::{
    AlertManager,
    AlertRule,
    AlertCondition,
    AlertChannel,
    AlertState,
    AlertNotifier,
    AlertSuppressionManager,
};

// Re-export health monitoring
pub use health_monitoring::{
    HealthChecker,
    SystemHealth,
    ComponentHealth,
    HealthStatus,
    HealthIssue,
    HealthMonitor,
    HealthEvaluator,
};

// Re-export performance analysis
pub use performance_analysis::{
    PerformanceAnalyzer,
    PerformanceInsights,
    AnomalyDetector,
    TrendAnalyzer,
    CorrelationAnalyzer,
    Insight,
    Pattern,
    Anomaly,
    Correlation,
    PerformanceTrends,
    TrendDirection,
    Recommendation,
};

// Re-export configuration management
pub use configuration_management::{
    MonitoringConfig,
    MetricsCollectionConfig,
    EventTrackingConfig,
    AlertConfig,
    HealthCheckConfig,
    PerformanceAnalysisConfig,
    AnomalyDetectionConfig,
    DataRetentionConfig,
    ExportConfig,
    ReportConfig,
    ConfigValidator,
    ConfigBuilder,
};

// Re-export reporting system
pub use reporting_system::{
    ReportGenerator,
    MonitoringReport,
    ReportSection,
    ReportFormat,
    ReportType,
    DetailLevel,
    ReportBuilder,
    ReportRenderer,
    ReportExporter,
};

// Re-export data retention
pub use data_retention::{
    DataRetentionManager,
    RetentionPolicy,
    ArchiveManager,
    DataExporter,
    CleanupManager,
    ArchiveConfig,
    CleanupConfig,
    ExportFormat,
    CompressionConfig,
};

// Legacy compatibility aliases for existing APIs
pub use monitoring_core::{
    ExecutionMonitor,
    DefaultExecutionMonitor,
};

pub use metrics_collection::{
    PerformanceMetric as Metric,
    MetricsStorage as Storage,
};

pub use session_management::{
    MonitoringSession as Session,
    SessionManager as SessionMgr,
};

// Convenience functions that maintain API compatibility
pub fn create_default_monitor() -> DefaultExecutionMonitor {
    let config = MonitoringConfig::default();
    DefaultExecutionMonitor::new(config)
}

pub fn create_configured_monitor(config: MonitoringConfig) -> DefaultExecutionMonitor {
    DefaultExecutionMonitor::new(config)
}

pub fn create_performance_optimized_monitor() -> DefaultExecutionMonitor {
    let config = MonitoringConfig::performance_optimized();
    DefaultExecutionMonitor::new(config)
}

pub fn create_minimal_overhead_monitor() -> DefaultExecutionMonitor {
    let config = MonitoringConfig::minimal_overhead();
    DefaultExecutionMonitor::new(config)
}

/// Execution monitoring framework version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const FRAMEWORK_NAME: &str = "Sklears Execution Monitoring";

/// Framework capabilities and feature flags
#[derive(Debug, Clone)]
pub struct MonitoringCapabilities {
    /// Real-time metrics collection
    pub real_time_metrics: bool,
    /// Advanced anomaly detection
    pub anomaly_detection: bool,
    /// Machine learning insights
    pub ml_insights: bool,
    /// Multi-session support
    pub multi_session: bool,
    /// Distributed monitoring
    pub distributed: bool,
    /// Custom extensibility
    pub extensible: bool,
}

impl Default for MonitoringCapabilities {
    fn default() -> Self {
        Self {
            real_time_metrics: true,
            anomaly_detection: true,
            ml_insights: true,
            multi_session: true,
            distributed: false, // Future feature
            extensible: true,
        }
    }
}

/// Get current framework capabilities
pub fn get_monitoring_capabilities() -> MonitoringCapabilities {
    MonitoringCapabilities::default()
}

/// Framework initialization and setup utilities
pub struct MonitoringFramework;

impl MonitoringFramework {
    /// Initialize the monitoring framework with default settings
    pub fn init() -> SklResult<()> {
        // Initialize logging, metrics backends, etc.
        log::info!("Initializing {} v{}", FRAMEWORK_NAME, VERSION);
        Ok(())
    }

    /// Initialize with custom configuration
    pub fn init_with_config(_config: MonitoringConfig) -> SklResult<()> {
        // Custom initialization logic
        Ok(())
    }

    /// Shutdown the monitoring framework gracefully
    pub fn shutdown() -> SklResult<()> {
        log::info!("Shutting down {} v{}", FRAMEWORK_NAME, VERSION);
        Ok(())
    }

    /// Get framework information
    pub fn info() -> FrameworkInfo {
        FrameworkInfo {
            name: FRAMEWORK_NAME.to_string(),
            version: VERSION.to_string(),
            capabilities: get_monitoring_capabilities(),
            build_date: env!("BUILD_DATE").to_string(),
            git_hash: env!("GIT_HASH").to_string(),
        }
    }
}

/// Framework information structure
#[derive(Debug, Clone)]
pub struct FrameworkInfo {
    pub name: String,
    pub version: String,
    pub capabilities: MonitoringCapabilities,
    pub build_date: String,
    pub git_hash: String,
}

/// Global monitoring registry for managing multiple monitors
#[derive(Debug)]
pub struct MonitoringRegistry {
    monitors: HashMap<String, Box<dyn ExecutionMonitor>>,
}

impl MonitoringRegistry {
    /// Create new monitoring registry
    pub fn new() -> Self {
        Self {
            monitors: HashMap::new(),
        }
    }

    /// Register a new monitor
    pub fn register_monitor(&mut self, name: String, monitor: Box<dyn ExecutionMonitor>) {
        self.monitors.insert(name, monitor);
    }

    /// Get monitor by name
    pub fn get_monitor(&self, name: &str) -> Option<&dyn ExecutionMonitor> {
        self.monitors.get(name).map(|m| m.as_ref())
    }

    /// Get mutable monitor by name
    pub fn get_monitor_mut(&mut self, name: &str) -> Option<&mut dyn ExecutionMonitor> {
        self.monitors.get_mut(name).map(|m| m.as_mut())
    }

    /// List all registered monitors
    pub fn list_monitors(&self) -> Vec<&str> {
        self.monitors.keys().map(|k| k.as_str()).collect()
    }

    /// Remove monitor by name
    pub fn unregister_monitor(&mut self, name: &str) -> Option<Box<dyn ExecutionMonitor>> {
        self.monitors.remove(name)
    }
}

impl Default for MonitoringRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global monitoring utilities
pub mod utils {
    use super::*;

    /// Generate unique session ID
    pub fn generate_session_id() -> String {
        uuid::Uuid::new_v4().to_string()
    }

    /// Get current timestamp
    pub fn current_timestamp() -> SystemTime {
        SystemTime::now()
    }

    /// Calculate duration between timestamps
    pub fn calculate_duration(start: SystemTime, end: SystemTime) -> Duration {
        end.duration_since(start).unwrap_or(Duration::from_secs(0))
    }

    /// Format duration for human-readable display
    pub fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs();
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
        }
    }

    /// Validate session ID format
    pub fn validate_session_id(session_id: &str) -> bool {
        !session_id.is_empty() && session_id.len() <= 255
    }

    /// Create test monitoring configuration
    pub fn create_test_config() -> MonitoringConfig {
        MonitoringConfig::test_config()
    }

    /// Create development monitoring configuration
    pub fn create_dev_config() -> MonitoringConfig {
        MonitoringConfig::development()
    }

    /// Create production monitoring configuration
    pub fn create_production_config() -> MonitoringConfig {
        MonitoringConfig::production()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_info() {
        let info = MonitoringFramework::info();
        assert_eq!(info.name, FRAMEWORK_NAME);
        assert_eq!(info.version, VERSION);
        assert!(info.capabilities.real_time_metrics);
    }

    #[test]
    fn test_monitoring_registry() {
        let mut registry = MonitoringRegistry::new();
        assert_eq!(registry.list_monitors().len(), 0);

        let config = MonitoringConfig::default();
        let monitor = Box::new(DefaultExecutionMonitor::new(config));
        registry.register_monitor("test_monitor".to_string(), monitor);

        assert_eq!(registry.list_monitors().len(), 1);
        assert!(registry.get_monitor("test_monitor").is_some());
    }

    #[test]
    fn test_utils() {
        let session_id = utils::generate_session_id();
        assert!(utils::validate_session_id(&session_id));

        let start = utils::current_timestamp();
        std::thread::sleep(Duration::from_millis(10));
        let end = utils::current_timestamp();

        let duration = utils::calculate_duration(start, end);
        assert!(duration.as_millis() >= 10);

        let formatted = utils::format_duration(Duration::from_secs(125));
        assert_eq!(formatted, "2m 5s");
    }

    #[test]
    fn test_convenience_functions() {
        let _default_monitor = create_default_monitor();
        let _optimized_monitor = create_performance_optimized_monitor();
        let _minimal_monitor = create_minimal_overhead_monitor();

        let config = MonitoringConfig::default();
        let _configured_monitor = create_configured_monitor(config);
    }

    #[test]
    fn test_capabilities() {
        let capabilities = get_monitoring_capabilities();
        assert!(capabilities.real_time_metrics);
        assert!(capabilities.anomaly_detection);
        assert!(capabilities.ml_insights);
        assert!(capabilities.multi_session);
        assert!(capabilities.extensible);
    }
}