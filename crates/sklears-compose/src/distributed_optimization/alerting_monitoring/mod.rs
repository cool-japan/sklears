//! Comprehensive Alerting and Monitoring System for Distributed Optimization
//!
//! This module provides a complete alerting and monitoring solution for distributed systems,
//! including alert management, notifications, escalation policies, event correlation,
//! dashboards, metrics collection, data persistence, and real-time monitoring.
//!
//! # Architecture
//!
//! The alerting and monitoring system is built with a modular architecture that separates
//! concerns into focused, maintainable components:
//!
//! ## Core Components
//!
//! - **Alert Management**: Rule configuration, state tracking, and evaluation engine
//! - **Notification Channels**: Multi-channel delivery with authentication and health monitoring
//! - **Escalation Policies**: Hierarchical escalation with sophisticated workflow management
//! - **Correlation Engine**: Intelligent event correlation and pattern recognition
//! - **Monitoring Dashboard**: Real-time visualization and interactive user interfaces
//! - **Metrics Collection**: Comprehensive metrics gathering and statistical analysis
//! - **Data Persistence**: Storage, archival, backup, and recovery systems
//! - **Real-Time Monitoring**: Live streaming, analytics, and event processing
//!
//! # Quick Start
//!
//! ```rust
//! use sklears_compose::distributed_optimization::alerting_monitoring::*;
//!
//! // Create alerting and monitoring system
//! let config = AlertingMonitoringConfig::default();
//! let system = AlertingMonitoringSystem::new(config)?;
//!
//! // Create an alert rule
//! let alert_rule = AlertRule::builder()
//!     .name("High CPU Usage")
//!     .condition(AlertCondition::metric_threshold("cpu_usage", ">", 80.0))
//!     .severity(AlertSeverity::Warning)
//!     .build();
//!
//! system.add_alert_rule(alert_rule)?;
//!
//! // Set up notification channel
//! let email_channel = NotificationChannel::email()
//!     .smtp_server("smtp.example.com")
//!     .recipients(vec!["admin@example.com"])
//!     .build();
//!
//! system.add_notification_channel(email_channel)?;
//!
//! // Start monitoring
//! system.start_monitoring()?;
//! ```
//!
//! # Advanced Features
//!
//! ## Event Correlation
//!
//! The correlation engine provides sophisticated pattern recognition:
//!
//! ```rust
//! use sklears_compose::distributed_optimization::alerting_monitoring::correlation_engine::*;
//!
//! let correlation_rule = CorrelationRule::builder()
//!     .name("Service Cascade Failure")
//!     .pattern(CorrelationPattern::temporal()
//!         .time_window(Duration::from_minutes(5))
//!         .sequence_requirement(SequenceRequirement::Flexible))
//!     .action(CorrelationAction::create_incident())
//!     .build();
//! ```
//!
//! ## Real-Time Analytics
//!
//! Stream processing with real-time analytics:
//!
//! ```rust
//! use sklears_compose::distributed_optimization::alerting_monitoring::real_time_monitoring::*;
//!
//! let stream_config = StreamConfiguration::builder()
//!     .stream_type(StreamType::MetricStream)
//!     .protocol(ProtocolType::WebSocket)
//!     .analytics(RealTimeAnalyticsConfig::builder()
//!         .anomaly_detection_enabled(true)
//!         .pattern_detection_enabled(true)
//!         .build())
//!     .build();
//! ```
//!
//! ## Data Persistence
//!
//! Comprehensive backup and recovery:
//!
//! ```rust
//! use sklears_compose::distributed_optimization::alerting_monitoring::data_persistence::*;
//!
//! let backup_config = BackupConfiguration::builder()
//!     .strategy(BackupStrategyType::Incremental)
//!     .schedule(BackupSchedule::Daily)
//!     .encryption_enabled(true)
//!     .verification_enabled(true)
//!     .build();
//! ```

// Module declarations for all specialized components
pub mod alert_management;
pub mod notification_channels;
pub mod notification_performance;
pub mod escalation_policies;
pub mod correlation_engine;
pub mod monitoring_dashboard;
pub mod metrics_collection;
pub mod data_persistence;
pub mod real_time_monitoring;

// Re-export key types and traits for easy access
pub use alert_management::{
    AlertManager, AlertRule, AlertCondition, AlertSeverity, AlertState, AlertThreshold,
    AlertEvaluationEngine, AlertMetrics, AlertHistoryEntry, AlertRule as Alert,
};

pub use notification_channels::{
    NotificationChannelManager, NotificationChannel, ChannelType, DeliveryStatus,
    MessageTemplate, ChannelConfig, AuthenticationMode, HealthMonitor as ChannelHealthMonitor,
};

pub use notification_performance::{
    PerformanceManager, ChannelPerformanceConfig, CompressionAlgorithm, ConnectionManager,
    CacheManager, CompressionManager, PerformanceOptimizer, PerformanceMonitor, PerformanceLoadBalancer,
};

pub use escalation_policies::{
    EscalationPolicyManager, EscalationPolicy, EscalationLevel, EscalationAction,
    EscalationTrigger, EscalationExecution, EscalationStatus, EscalationMetrics,
};

pub use correlation_engine::{
    CorrelationEngine, CorrelationRule, CorrelationPattern, CorrelatedEventGroup,
    TemporalPattern, SpatialPattern, CausalPattern, StatisticalPattern, BehavioralPattern,
};

pub use monitoring_dashboard::{
    DashboardManager, Dashboard, Widget, WidgetType, ChartConfiguration, DashboardTheme,
    DataSourceConfiguration, RealTimeHandler, RenderingEngine,
};

pub use metrics_collection::{
    MetricsCollectionManager, MetricDefinition, MetricType, MetricDataPoint, MetricStatistics,
    CollectionConfiguration, AggregationConfiguration, RetentionConfiguration,
};

pub use data_persistence::{
    DataPersistenceManager, StorageBackend, BackupConfiguration, RecoveryManager,
    ArchivalManager, BackupJob, BackupJobStatus, RecoveryPlan, StorageBackendType,
};

pub use real_time_monitoring::{
    RealTimeMonitoringManager, RealTimeEvent, StreamConfiguration, StreamSubscription,
    EventType, EventPriority, StreamProcessor, ConnectionManager, AnalyticsEngine,
};

// Common error types
use std::fmt;
use thiserror::Error;

/// Comprehensive error type for the alerting and monitoring system
#[derive(Error, Debug)]
pub enum AlertingMonitoringError {
    #[error("Alert management error: {0}")]
    AlertManagement(#[from] alert_management::AlertError),
    #[error("Notification error: {0}")]
    Notification(#[from] notification_channels::NotificationError),
    #[error("Escalation error: {0}")]
    Escalation(#[from] escalation_policies::EscalationError),
    #[error("Correlation error: {0}")]
    Correlation(#[from] correlation_engine::CorrelationError),
    #[error("Dashboard error: {0}")]
    Dashboard(#[from] monitoring_dashboard::DashboardError),
    #[error("Metrics error: {0}")]
    Metrics(#[from] metrics_collection::MetricsError),
    #[error("Persistence error: {0}")]
    Persistence(#[from] data_persistence::PersistenceError),
    #[error("Real-time monitoring error: {0}")]
    RealTime(#[from] real_time_monitoring::RealTimeError),
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("System error: {0}")]
    System(String),
}

/// Result type for alerting and monitoring operations
pub type AlertingMonitoringResult<T> = Result<T, AlertingMonitoringError>;

// Common types and traits
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};

/// Configuration for the entire alerting and monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingMonitoringConfig {
    /// Alert management configuration
    pub alert_management: alert_management::AlertManagerConfig,
    /// Notification channels configuration
    pub notification_channels: notification_channels::ChannelManagerConfig,
    /// Escalation policies configuration
    pub escalation_policies: escalation_policies::EscalationPolicyConfig,
    /// Correlation engine configuration
    pub correlation_engine: correlation_engine::CorrelationEngineConfig,
    /// Dashboard configuration
    pub monitoring_dashboard: monitoring_dashboard::DashboardConfiguration,
    /// Metrics collection configuration
    pub metrics_collection: metrics_collection::MetricsCollectionConfig,
    /// Data persistence configuration
    pub data_persistence: data_persistence::BackupConfiguration,
    /// Real-time monitoring configuration
    pub real_time_monitoring: real_time_monitoring::StreamConfiguration,
    /// Global system settings
    pub global_settings: GlobalSettings,
}

/// Global settings for the alerting and monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSettings {
    /// System name for identification
    pub system_name: String,
    /// Environment (dev, staging, prod, etc.)
    pub environment: String,
    /// Default timezone for all operations
    pub default_timezone: String,
    /// Maximum memory usage for the system
    pub max_memory_usage: u64,
    /// Maximum concurrent operations
    pub max_concurrent_operations: u32,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Performance monitoring enabled
    pub performance_monitoring_enabled: bool,
    /// Debug mode enabled
    pub debug_mode: bool,
    /// Feature flags for experimental features
    pub feature_flags: HashMap<String, bool>,
}

impl Default for AlertingMonitoringConfig {
    fn default() -> Self {
        Self {
            alert_management: alert_management::AlertManagerConfig::default(),
            notification_channels: notification_channels::ChannelManagerConfig::default(),
            escalation_policies: escalation_policies::EscalationPolicyConfig::default(),
            correlation_engine: correlation_engine::CorrelationEngineConfig::default(),
            monitoring_dashboard: monitoring_dashboard::DashboardConfiguration::default(),
            metrics_collection: metrics_collection::MetricsCollectionConfig::default(),
            data_persistence: data_persistence::BackupConfiguration::default(),
            real_time_monitoring: real_time_monitoring::StreamConfiguration::default(),
            global_settings: GlobalSettings::default(),
        }
    }
}

impl Default for GlobalSettings {
    fn default() -> Self {
        Self {
            system_name: "AlertingMonitoringSystem".to_string(),
            environment: "development".to_string(),
            default_timezone: "UTC".to_string(),
            max_memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
            max_concurrent_operations: 1000,
            health_check_interval: Duration::from_secs(30),
            performance_monitoring_enabled: true,
            debug_mode: false,
            feature_flags: HashMap::new(),
        }
    }
}

/// System status for monitoring the health of the alerting and monitoring system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemStatus {
    /// System is healthy and operational
    Healthy,
    /// System is operational but with warnings
    Warning,
    /// System has critical issues but is still operational
    Critical,
    /// System is not operational
    Down,
    /// System status is unknown
    Unknown,
}

/// Performance metrics for the alerting and monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformanceMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Number of active alerts
    pub active_alerts: u32,
    /// Number of processed events per second
    pub events_per_second: f64,
    /// Average response time for operations
    pub average_response_time: Duration,
    /// Number of active connections
    pub active_connections: u32,
    /// Storage utilization percentage
    pub storage_utilization: f64,
    /// Network throughput in bytes per second
    pub network_throughput: f64,
}

/// Health check result for system components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Component name
    pub component: String,
    /// Health status
    pub status: SystemStatus,
    /// Last check timestamp
    pub last_check: SystemTime,
    /// Response time for the health check
    pub response_time: Duration,
    /// Additional details about the health check
    pub details: HashMap<String, String>,
    /// Error message if the check failed
    pub error_message: Option<String>,
}

/// Main alerting and monitoring system that coordinates all components
pub struct AlertingMonitoringSystem {
    /// System configuration
    config: AlertingMonitoringConfig,
    /// Alert management subsystem
    alert_manager: Arc<RwLock<alert_management::AlertManager>>,
    /// Notification channel manager
    notification_manager: Arc<RwLock<notification_channels::NotificationChannelManager>>,
    /// Escalation policy manager
    escalation_manager: Arc<RwLock<escalation_policies::EscalationPolicyManager>>,
    /// Event correlation engine
    correlation_engine: Arc<RwLock<correlation_engine::CorrelationEngine>>,
    /// Dashboard manager
    dashboard_manager: Arc<RwLock<monitoring_dashboard::DashboardManager>>,
    /// Metrics collection manager
    metrics_manager: Arc<RwLock<metrics_collection::MetricsCollectionManager>>,
    /// Data persistence manager
    persistence_manager: Arc<RwLock<data_persistence::DataPersistenceManager>>,
    /// Real-time monitoring manager
    realtime_manager: Arc<RwLock<real_time_monitoring::RealTimeMonitoringManager>>,
    /// System status
    system_status: Arc<RwLock<SystemStatus>>,
    /// Performance metrics
    performance_metrics: Arc<RwLock<SystemPerformanceMetrics>>,
    /// Health check results
    health_check_results: Arc<RwLock<HashMap<String, HealthCheckResult>>>,
}

impl AlertingMonitoringSystem {
    /// Create a new alerting and monitoring system
    pub fn new(config: AlertingMonitoringConfig) -> AlertingMonitoringResult<Self> {
        // Initialize all subsystems
        let alert_manager = Arc::new(RwLock::new(
            alert_management::AlertManager::new(config.alert_management.clone())
        ));

        let notification_manager = Arc::new(RwLock::new(
            notification_channels::NotificationChannelManager::new(config.notification_channels.clone())
        ));

        let escalation_manager = Arc::new(RwLock::new(
            escalation_policies::EscalationPolicyManager::new(config.escalation_policies.clone())
        ));

        let correlation_engine = Arc::new(RwLock::new(
            correlation_engine::CorrelationEngine::new(config.correlation_engine.clone())
        ));

        let dashboard_manager = Arc::new(RwLock::new(
            monitoring_dashboard::DashboardManager::new(config.monitoring_dashboard.clone())
        ));

        let metrics_manager = Arc::new(RwLock::new(
            metrics_collection::MetricsCollectionManager::new(config.metrics_collection.clone())
        ));

        let persistence_manager = Arc::new(RwLock::new(
            data_persistence::DataPersistenceManager::new()
        ));

        let realtime_manager = Arc::new(RwLock::new(
            real_time_monitoring::RealTimeMonitoringManager::new()
        ));

        // Initialize system metrics
        let performance_metrics = SystemPerformanceMetrics {
            cpu_usage: 0.0,
            memory_usage: 0,
            active_alerts: 0,
            events_per_second: 0.0,
            average_response_time: Duration::from_millis(0),
            active_connections: 0,
            storage_utilization: 0.0,
            network_throughput: 0.0,
        };

        Ok(Self {
            config,
            alert_manager,
            notification_manager,
            escalation_manager,
            correlation_engine,
            dashboard_manager,
            metrics_manager,
            persistence_manager,
            realtime_manager,
            system_status: Arc::new(RwLock::new(SystemStatus::Unknown)),
            performance_metrics: Arc::new(RwLock::new(performance_metrics)),
            health_check_results: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start the alerting and monitoring system
    pub fn start(&self) -> AlertingMonitoringResult<()> {
        // Update system status
        *self.system_status.write().unwrap() = SystemStatus::Healthy;

        // Start all subsystems
        self.start_alert_management()?;
        self.start_notification_channels()?;
        self.start_escalation_policies()?;
        self.start_correlation_engine()?;
        self.start_dashboard()?;
        self.start_metrics_collection()?;
        self.start_data_persistence()?;
        self.start_real_time_monitoring()?;

        // Start health monitoring
        self.start_health_monitoring()?;

        // Start performance monitoring
        if self.config.global_settings.performance_monitoring_enabled {
            self.start_performance_monitoring()?;
        }

        Ok(())
    }

    /// Stop the alerting and monitoring system
    pub fn stop(&self) -> AlertingMonitoringResult<()> {
        *self.system_status.write().unwrap() = SystemStatus::Down;

        // Stop all subsystems gracefully
        // Implementation would stop all managers and clean up resources

        Ok(())
    }

    /// Get the current system status
    pub fn get_system_status(&self) -> SystemStatus {
        self.system_status.read().unwrap().clone()
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> SystemPerformanceMetrics {
        self.performance_metrics.read().unwrap().clone()
    }

    /// Get health check results for all components
    pub fn get_health_check_results(&self) -> HashMap<String, HealthCheckResult> {
        self.health_check_results.read().unwrap().clone()
    }

    /// Add an alert rule to the system
    pub fn add_alert_rule(&self, rule: alert_management::AlertRule) -> AlertingMonitoringResult<()> {
        let alert_manager = self.alert_manager.write().unwrap();
        alert_manager.add_alert_rule(rule)?;
        Ok(())
    }

    /// Add a notification channel
    pub fn add_notification_channel(&self, channel: notification_channels::NotificationChannel) -> AlertingMonitoringResult<()> {
        let notification_manager = self.notification_manager.write().unwrap();
        notification_manager.add_channel(channel)?;
        Ok(())
    }

    /// Add an escalation policy
    pub fn add_escalation_policy(&self, policy: escalation_policies::EscalationPolicy) -> AlertingMonitoringResult<()> {
        let escalation_manager = self.escalation_manager.write().unwrap();
        escalation_manager.add_policy(policy)?;
        Ok(())
    }

    /// Add a correlation rule
    pub fn add_correlation_rule(&self, rule: correlation_engine::CorrelationRule) -> AlertingMonitoringResult<()> {
        let correlation_engine = self.correlation_engine.write().unwrap();
        correlation_engine.add_rule(rule)?;
        Ok(())
    }

    /// Create a dashboard
    pub fn create_dashboard(&self, dashboard: monitoring_dashboard::Dashboard) -> AlertingMonitoringResult<()> {
        let dashboard_manager = self.dashboard_manager.write().unwrap();
        dashboard_manager.create_dashboard(dashboard)?;
        Ok(())
    }

    /// Register a metric definition
    pub fn register_metric(&self, metric: metrics_collection::MetricDefinition) -> AlertingMonitoringResult<()> {
        let metrics_manager = self.metrics_manager.write().unwrap();
        metrics_manager.register_metric(metric)?;
        Ok(())
    }

    /// Create a backup configuration
    pub fn create_backup_config(&self, config: data_persistence::BackupConfiguration) -> AlertingMonitoringResult<String> {
        let persistence_manager = self.persistence_manager.write().unwrap();
        let config_id = persistence_manager.create_backup_config(config)?;
        Ok(config_id)
    }

    /// Create a real-time stream
    pub fn create_stream(&self, config: real_time_monitoring::StreamConfiguration) -> AlertingMonitoringResult<()> {
        let realtime_manager = self.realtime_manager.write().unwrap();
        realtime_manager.create_stream(config)?;
        Ok(())
    }

    /// Process an incoming event
    pub fn process_event(&self, event: correlation_engine::CorrelationEvent) -> AlertingMonitoringResult<()> {
        // Send event to correlation engine
        let correlation_engine = self.correlation_engine.write().unwrap();
        correlation_engine.process_event(event)?;

        // Update performance metrics
        self.update_performance_metrics();

        Ok(())
    }

    /// Get alerts by status
    pub fn get_alerts_by_status(&self, status: alert_management::AlertState) -> AlertingMonitoringResult<Vec<alert_management::ActiveAlert>> {
        let alert_manager = self.alert_manager.read().unwrap();
        let alerts = alert_manager.get_alerts_by_status(status)?;
        Ok(alerts)
    }

    /// Get system statistics
    pub fn get_system_statistics(&self) -> AlertingMonitoringResult<SystemStatistics> {
        let alert_manager = self.alert_manager.read().unwrap();
        let notification_manager = self.notification_manager.read().unwrap();
        let metrics_manager = self.metrics_manager.read().unwrap();

        let stats = SystemStatistics {
            total_alerts: alert_manager.get_total_alert_count()?,
            active_alerts: alert_manager.get_active_alert_count()?,
            total_notifications_sent: notification_manager.get_total_notifications_sent()?,
            total_metrics_collected: metrics_manager.get_total_metrics_collected()?,
            system_uptime: self.get_system_uptime(),
            last_backup: self.get_last_backup_time()?,
            storage_usage: self.get_storage_usage()?,
        };

        Ok(stats)
    }

    // Private helper methods

    fn start_alert_management(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start the alert management subsystem
        Ok(())
    }

    fn start_notification_channels(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start the notification channel subsystem
        Ok(())
    }

    fn start_escalation_policies(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start the escalation policy subsystem
        Ok(())
    }

    fn start_correlation_engine(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start the correlation engine
        Ok(())
    }

    fn start_dashboard(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start the dashboard manager
        Ok(())
    }

    fn start_metrics_collection(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start the metrics collection subsystem
        Ok(())
    }

    fn start_data_persistence(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start the data persistence subsystem
        Ok(())
    }

    fn start_real_time_monitoring(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start the real-time monitoring subsystem
        Ok(())
    }

    fn start_health_monitoring(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start periodic health checks
        Ok(())
    }

    fn start_performance_monitoring(&self) -> AlertingMonitoringResult<()> {
        // Implementation would start performance metric collection
        Ok(())
    }

    fn update_performance_metrics(&self) {
        // Implementation would update the performance metrics
        let mut metrics = self.performance_metrics.write().unwrap();
        metrics.events_per_second += 1.0; // Simplified example
    }

    fn get_system_uptime(&self) -> Duration {
        // Implementation would calculate system uptime
        Duration::from_secs(0)
    }

    fn get_last_backup_time(&self) -> AlertingMonitoringResult<Option<SystemTime>> {
        // Implementation would get the last backup time
        Ok(None)
    }

    fn get_storage_usage(&self) -> AlertingMonitoringResult<u64> {
        // Implementation would calculate storage usage
        Ok(0)
    }
}

/// System statistics for monitoring the alerting and monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatistics {
    /// Total number of alerts in the system
    pub total_alerts: u64,
    /// Number of currently active alerts
    pub active_alerts: u32,
    /// Total number of notifications sent
    pub total_notifications_sent: u64,
    /// Total number of metrics collected
    pub total_metrics_collected: u64,
    /// System uptime
    pub system_uptime: Duration,
    /// Last backup timestamp
    pub last_backup: Option<SystemTime>,
    /// Storage usage in bytes
    pub storage_usage: u64,
}

/// Builder pattern for creating AlertingMonitoringConfig
pub struct AlertingMonitoringConfigBuilder {
    config: AlertingMonitoringConfig,
}

impl AlertingMonitoringConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: AlertingMonitoringConfig::default(),
        }
    }

    /// Set the system name
    pub fn system_name(mut self, name: impl Into<String>) -> Self {
        self.config.global_settings.system_name = name.into();
        self
    }

    /// Set the environment
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.config.global_settings.environment = env.into();
        self
    }

    /// Set the default timezone
    pub fn timezone(mut self, tz: impl Into<String>) -> Self {
        self.config.global_settings.default_timezone = tz.into();
        self
    }

    /// Set maximum memory usage
    pub fn max_memory_usage(mut self, bytes: u64) -> Self {
        self.config.global_settings.max_memory_usage = bytes;
        self
    }

    /// Enable debug mode
    pub fn debug_mode(mut self, enabled: bool) -> Self {
        self.config.global_settings.debug_mode = enabled;
        self
    }

    /// Enable performance monitoring
    pub fn performance_monitoring(mut self, enabled: bool) -> Self {
        self.config.global_settings.performance_monitoring_enabled = enabled;
        self
    }

    /// Set a feature flag
    pub fn feature_flag(mut self, flag: impl Into<String>, enabled: bool) -> Self {
        self.config.global_settings.feature_flags.insert(flag.into(), enabled);
        self
    }

    /// Configure alert management
    pub fn alert_management(mut self, config: alert_management::AlertManagerConfig) -> Self {
        self.config.alert_management = config;
        self
    }

    /// Configure notification channels
    pub fn notification_channels(mut self, config: notification_channels::ChannelManagerConfig) -> Self {
        self.config.notification_channels = config;
        self
    }

    /// Configure escalation policies
    pub fn escalation_policies(mut self, config: escalation_policies::EscalationPolicyConfig) -> Self {
        self.config.escalation_policies = config;
        self
    }

    /// Configure correlation engine
    pub fn correlation_engine(mut self, config: correlation_engine::CorrelationEngineConfig) -> Self {
        self.config.correlation_engine = config;
        self
    }

    /// Configure monitoring dashboard
    pub fn monitoring_dashboard(mut self, config: monitoring_dashboard::DashboardConfiguration) -> Self {
        self.config.monitoring_dashboard = config;
        self
    }

    /// Configure metrics collection
    pub fn metrics_collection(mut self, config: metrics_collection::MetricsCollectionConfig) -> Self {
        self.config.metrics_collection = config;
        self
    }

    /// Configure data persistence
    pub fn data_persistence(mut self, config: data_persistence::BackupConfiguration) -> Self {
        self.config.data_persistence = config;
        self
    }

    /// Configure real-time monitoring
    pub fn real_time_monitoring(mut self, config: real_time_monitoring::StreamConfiguration) -> Self {
        self.config.real_time_monitoring = config;
        self
    }

    /// Build the configuration
    pub fn build(self) -> AlertingMonitoringConfig {
        self.config
    }
}

impl Default for AlertingMonitoringConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Utility functions for common operations

/// Create a simple alert rule for metric thresholds
pub fn create_metric_threshold_alert(
    name: impl Into<String>,
    metric_name: impl Into<String>,
    threshold: f64,
    operator: impl Into<String>,
    severity: alert_management::AlertSeverity,
) -> alert_management::AlertRule {
    alert_management::AlertRule {
        rule_id: format!("alert_{}", uuid::Uuid::new_v4()),
        name: name.into(),
        description: format!("Alert for {} {} {}", metric_name.into(), operator.into(), threshold),
        enabled: true,
        conditions: vec![alert_management::AlertCondition::MetricThreshold {
            metric: metric_name.into(),
            operator: operator.into(),
            threshold,
            duration: Duration::from_minutes(5),
        }],
        severity,
        tags: HashMap::new(),
        metadata: alert_management::AlertMetadata {
            created_at: SystemTime::now(),
            created_by: "system".to_string(),
            modified_at: SystemTime::now(),
            modified_by: "system".to_string(),
            version: "1.0".to_string(),
        },
        evaluation_config: alert_management::EvaluationConfig::default(),
        notification_config: alert_management::NotificationConfig::default(),
    }
}

/// Create a simple email notification channel
pub fn create_email_notification_channel(
    name: impl Into<String>,
    smtp_server: impl Into<String>,
    recipients: Vec<String>,
) -> notification_channels::NotificationChannel {
    notification_channels::NotificationChannel {
        channel_id: format!("channel_{}", uuid::Uuid::new_v4()),
        name: name.into(),
        channel_type: notification_channels::ChannelType::Email,
        enabled: true,
        configuration: notification_channels::ChannelConfig::Email {
            smtp_server: smtp_server.into(),
            port: 587,
            username: "".to_string(),
            password: "".to_string(),
            from_address: "noreply@example.com".to_string(),
            to_addresses: recipients,
            use_tls: true,
        },
        authentication: notification_channels::AuthenticationConfig::default(),
        rate_limiting: notification_channels::RateLimitingConfig::default(),
        health_check: notification_channels::HealthCheckConfig::default(),
        templates: HashMap::new(),
        metadata: notification_channels::ChannelMetadata::default(),
    }
}

/// Create a simple dashboard with basic widgets
pub fn create_basic_dashboard(
    name: impl Into<String>,
    description: impl Into<String>,
) -> monitoring_dashboard::Dashboard {
    monitoring_dashboard::Dashboard {
        dashboard_id: format!("dashboard_{}", uuid::Uuid::new_v4()),
        name: name.into(),
        description: Some(description.into()),
        dashboard_type: monitoring_dashboard::DashboardType::System,
        priority: monitoring_dashboard::DashboardPriority::Normal,
        visibility: monitoring_dashboard::VisibilityLevel::Internal,
        layout: monitoring_dashboard::DashboardLayout {
            layout_type: monitoring_dashboard::LayoutType::Grid,
            grid_config: monitoring_dashboard::GridConfiguration {
                columns: 12,
                rows: None,
                column_gap: 10.0,
                row_gap: 10.0,
                auto_rows: true,
                auto_columns: false,
                grid_template: None,
            },
            responsive_config: monitoring_dashboard::ResponsiveConfiguration {
                enabled: true,
                breakpoints: Vec::new(),
                adaptation_strategy: monitoring_dashboard::AdaptationStrategy::Scale,
                reflow_enabled: true,
            },
            container_config: monitoring_dashboard::ContainerConfiguration {
                max_width: None,
                max_height: None,
                padding: monitoring_dashboard::PaddingConfiguration {
                    top: 10.0,
                    right: 10.0,
                    bottom: 10.0,
                    left: 10.0,
                },
                margin: monitoring_dashboard::PaddingConfiguration {
                    top: 0.0,
                    right: 0.0,
                    bottom: 0.0,
                    left: 0.0,
                },
                overflow: monitoring_dashboard::OverflowBehavior::Auto,
                scrolling: monitoring_dashboard::ScrollingConfiguration {
                    horizontal_scrolling: false,
                    vertical_scrolling: true,
                    smooth_scrolling: true,
                    scroll_bars: monitoring_dashboard::ScrollBarConfiguration {
                        visible: true,
                        width: 8.0,
                        color: "#cccccc".to_string(),
                        background_color: "#f5f5f5".to_string(),
                        style: monitoring_dashboard::ScrollBarStyle::Default,
                    },
                },
            },
        },
        theme: "default".to_string(),
        widgets: Vec::new(),
        data_sources: Vec::new(),
        filters: Vec::new(),
        variables: Vec::new(),
        time_range: monitoring_dashboard::TimeRangeConfiguration {
            default_range: monitoring_dashboard::TimeRange {
                start_time: SystemTime::now() - Duration::from_hours(1),
                end_time: SystemTime::now(),
                granularity: Duration::from_minutes(1),
                timezone: "UTC".to_string(),
            },
            available_ranges: Vec::new(),
            custom_range_enabled: true,
            auto_refresh: true,
            timezone_handling: monitoring_dashboard::TimezoneHandling::UTC,
        },
        refresh_config: monitoring_dashboard::DashboardRefreshConfig {
            auto_refresh: true,
            refresh_interval: Duration::from_seconds(30),
            refresh_on_focus: true,
            refresh_on_data_change: true,
            partial_refresh: true,
            background_refresh: true,
        },
        sharing_config: monitoring_dashboard::SharingConfiguration {
            enabled: false,
            share_types: Vec::new(),
            expiration_config: monitoring_dashboard::ExpirationConfig {
                enabled: false,
                default_expiration: Duration::from_hours(24),
                max_expiration: Duration::from_days(7),
                auto_extend: false,
            },
            password_protection: false,
            watermark_enabled: false,
            download_enabled: true,
        },
        permissions: monitoring_dashboard::DashboardPermissions {
            owner: "system".to_string(),
            view_permissions: vec!["all".to_string()],
            edit_permissions: vec!["admin".to_string()],
            admin_permissions: vec!["admin".to_string()],
            share_permissions: vec!["admin".to_string()],
            delete_permissions: vec!["admin".to_string()],
            inherit_permissions: true,
        },
        metadata: monitoring_dashboard::DashboardMetadata {
            created_at: SystemTime::now(),
            created_by: "system".to_string(),
            modified_at: SystemTime::now(),
            modified_by: "system".to_string(),
            version: "1.0".to_string(),
            tags: Vec::new(),
            category: "system".to_string(),
            starred_by: Vec::new(),
            usage_statistics: monitoring_dashboard::DashboardUsageStatistics {
                view_count: 0,
                unique_viewers: 0,
                average_session_duration: Duration::from_seconds(0),
                total_session_time: Duration::from_seconds(0),
                last_accessed: None,
                access_pattern: monitoring_dashboard::AccessPattern {
                    hourly_distribution: vec![0; 24],
                    daily_distribution: vec![0; 7],
                    weekly_distribution: vec![0; 52],
                    monthly_distribution: vec![0; 12],
                    peak_usage_times: Vec::new(),
                },
                popular_widgets: Vec::new(),
                popular_filters: Vec::new(),
            },
            performance_metrics: monitoring_dashboard::DashboardPerformanceMetrics {
                load_time: Duration::from_millis(0),
                render_time: Duration::from_millis(0),
                data_fetch_time: Duration::from_millis(0),
                memory_usage: 0,
                network_usage: 0,
                cpu_usage: 0.0,
                error_rate: 0.0,
                availability: 100.0,
            },
        },
    }
}

// Module-level tests
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = AlertingMonitoringConfigBuilder::new()
            .system_name("TestSystem")
            .environment("test")
            .debug_mode(true)
            .performance_monitoring(true)
            .feature_flag("experimental_feature", true)
            .build();

        assert_eq!(config.global_settings.system_name, "TestSystem");
        assert_eq!(config.global_settings.environment, "test");
        assert!(config.global_settings.debug_mode);
        assert!(config.global_settings.performance_monitoring_enabled);
        assert_eq!(config.global_settings.feature_flags.get("experimental_feature"), Some(&true));
    }

    #[test]
    fn test_system_creation() {
        let config = AlertingMonitoringConfig::default();
        let system = AlertingMonitoringSystem::new(config);
        assert!(system.is_ok());
    }

    #[test]
    fn test_metric_threshold_alert_creation() {
        let alert = create_metric_threshold_alert(
            "High CPU",
            "cpu_usage",
            80.0,
            ">",
            alert_management::AlertSeverity::Warning,
        );

        assert_eq!(alert.name, "High CPU");
        assert!(alert.enabled);
        assert_eq!(alert.severity, alert_management::AlertSeverity::Warning);
    }

    #[test]
    fn test_email_channel_creation() {
        let channel = create_email_notification_channel(
            "Admin Email",
            "smtp.example.com",
            vec!["admin@example.com".to_string()],
        );

        assert_eq!(channel.name, "Admin Email");
        assert!(channel.enabled);
        assert_eq!(channel.channel_type, notification_channels::ChannelType::Email);
    }

    #[test]
    fn test_basic_dashboard_creation() {
        let dashboard = create_basic_dashboard(
            "System Overview",
            "Main system monitoring dashboard",
        );

        assert_eq!(dashboard.name, "System Overview");
        assert_eq!(dashboard.dashboard_type, monitoring_dashboard::DashboardType::System);
        assert!(dashboard.widgets.is_empty());
    }
}

// Add necessary external dependencies to the uses
use uuid;
use chrono;