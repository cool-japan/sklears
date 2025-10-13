//! Configuration Builder and Health Checks
//!
//! This module contains builder patterns, validation logic, default implementations,
//! and health check configuration for the monitoring system. It provides a fluent
//! interface for constructing monitoring configurations and comprehensive validation.

use std::collections::HashMap;
use std::time::Duration;
use sklears_core::error::{Result as SklResult, SklearsError};

use super::config_core::MonitoringConfig;
use super::metrics_config::MetricsConfig;
use super::event_tracking::EventTrackingConfig;
use super::performance_config::PerformanceMonitoringConfig;
use super::resource_monitoring::ResourceMonitoringConfig;
use super::alert_configuration::AlertConfig;
use super::data_management::{DataRetentionConfig, ExportConfig, SamplingConfig};

/// Configuration builder for monitoring system
///
/// Provides a fluent interface for building monitoring configurations with
/// validation, default settings, and environment-specific optimizations.
///
/// # Builder Pattern Benefits
///
/// The builder pattern provides several advantages:
/// - **Fluent Interface**: Method chaining for readable configuration
/// - **Validation**: Early validation of configuration parameters
/// - **Defaults**: Sensible defaults for common use cases
/// - **Environment Profiles**: Pre-configured setups for different environments
/// - **Incremental Construction**: Step-by-step configuration building
///
/// # Usage Examples
///
/// ## Basic Configuration Building
/// ```rust
/// use sklears_compose::monitoring_config::ConfigBuilder;
///
/// let config = ConfigBuilder::new()
///     .enable_metrics(true)
///     .enable_alerts(true)
///     .sampling_rate(0.1)
///     .build()?;
/// ```
///
/// ## Environment-Specific Configuration
/// ```rust
/// // Production configuration
/// let prod_config = ConfigBuilder::production()
///     .with_custom_metric("api_requests")
///     .alert_channel("pagerduty")
///     .build()?;
///
/// // Development configuration
/// let dev_config = ConfigBuilder::development()
///     .enable_debug_logging(true)
///     .build()?;
/// ```
///
/// ## Advanced Configuration
/// ```rust
/// let config = ConfigBuilder::new()
///     .metrics_collection_interval(Duration::from_secs(30))
///     .event_buffer_size(10000)
///     .performance_profiling(true)
///     .resource_monitoring_all()
///     .export_format("json")
///     .retention_period(Duration::from_secs(86400 * 90))
///     .build()?;
/// ```
#[derive(Debug)]
pub struct ConfigBuilder {
    /// The configuration being built
    config: MonitoringConfig,
    /// Validation errors collected during building
    validation_errors: Vec<String>,
    /// Environment profile being used
    environment: EnvironmentProfile,
}

/// Environment profiles for different deployment scenarios
///
/// Provides pre-configured settings optimized for specific environments
/// and use cases.
#[derive(Debug, Clone)]
pub enum EnvironmentProfile {
    Development,
    Testing,
    Staging,
    Production,
    HighVolume,
    SecurityFocused,
    ComplianceFocused,
    Custom { name: String },
}

impl ConfigBuilder {
    /// Create a new configuration builder with default settings
    pub fn new() -> Self {
        Self {
            config: MonitoringConfig::default(),
            validation_errors: Vec::new(),
            environment: EnvironmentProfile::Development,
        }
    }

    /// Create a builder with production-optimized settings
    pub fn production() -> Self {
        Self {
            config: MonitoringConfig::production(),
            validation_errors: Vec::new(),
            environment: EnvironmentProfile::Production,
        }
    }

    /// Create a builder with development-optimized settings
    pub fn development() -> Self {
        Self {
            config: MonitoringConfig::development(),
            validation_errors: Vec::new(),
            environment: EnvironmentProfile::Development,
        }
    }

    /// Create a builder with testing-optimized settings
    pub fn testing() -> Self {
        Self {
            config: MonitoringConfig::testing(),
            validation_errors: Vec::new(),
            environment: EnvironmentProfile::Testing,
        }
    }

    /// Create a builder with high-volume production settings
    pub fn high_volume() -> Self {
        let mut config = MonitoringConfig::production();

        // Optimize for high volume
        config.sampling.enabled = true;
        config.sampling.rate = 0.01; // 1% sampling
        config.metrics.aggregation.real_time = true;
        config.metrics.storage.batch_size = 10000;
        config.events.collection.buffer_size = 50000;
        config.performance.profiling.enabled = false; // Reduce overhead

        Self {
            config,
            validation_errors: Vec::new(),
            environment: EnvironmentProfile::HighVolume,
        }
    }

    /// Create a builder with security-focused settings
    pub fn security_focused() -> Self {
        let mut config = MonitoringConfig::production();

        // Security-focused settings
        config.events = EventTrackingConfig::security_focused();
        config.alerts.enabled = true;
        config.export.enabled = false; // Reduce data exposure
        config.retention.events_retention = Duration::from_secs(86400 * 365 * 7); // 7 years

        Self {
            config,
            validation_errors: Vec::new(),
            environment: EnvironmentProfile::SecurityFocused,
        }
    }

    /// Create a builder with compliance-focused settings
    pub fn compliance_focused() -> Self {
        let mut config = MonitoringConfig::production();

        // Compliance settings
        config.retention.events_retention = Duration::from_secs(86400 * 365 * 10); // 10 years
        config.retention.archive.enabled = true;
        config.retention.archive.encryption.enabled = true;
        config.sampling.enabled = false; // Full data retention for compliance

        Self {
            config,
            validation_errors: Vec::new(),
            environment: EnvironmentProfile::ComplianceFocused,
        }
    }

    /// Set environment profile
    pub fn environment(mut self, profile: EnvironmentProfile) -> Self {
        self.environment = profile;
        self
    }

    // === Metrics Configuration ===

    /// Enable or disable metrics collection
    pub fn enable_metrics(mut self, enabled: bool) -> Self {
        self.config.metrics.enabled = enabled;
        self
    }

    /// Set metrics collection interval
    pub fn metrics_collection_interval(mut self, interval: Duration) -> Self {
        if interval < Duration::from_millis(100) {
            self.validation_errors.push("Metrics collection interval must be at least 100ms".to_string());
        }
        self.config.metrics.collection_interval = interval;
        self
    }

    /// Add custom metric to collection
    pub fn with_custom_metric(mut self, metric_name: &str) -> Self {
        use super::metrics_config::{MetricType, CustomMetricDefinition, CustomMetricType};

        self.config.metrics.metric_types.push(MetricType::Custom {
            name: metric_name.to_string(),
            description: format!("Custom metric: {}", metric_name),
        });

        self.config.metrics.custom_metrics.push(CustomMetricDefinition {
            name: metric_name.to_string(),
            description: format!("Custom metric: {}", metric_name),
            metric_type: CustomMetricType::Gauge,
            labels: Vec::new(),
            unit: None,
            help: None,
        });

        self
    }

    /// Configure metrics storage backend
    pub fn metrics_storage_backend(mut self, backend_type: &str) -> Self {
        use super::metrics_config::{StorageBackend, FileFormat};

        self.config.metrics.storage.backend = match backend_type {
            "memory" => StorageBackend::Memory,
            "file" => StorageBackend::File { format: FileFormat::Json },
            "binary" => StorageBackend::File { format: FileFormat::Binary },
            _ => {
                self.validation_errors.push(format!("Unknown storage backend: {}", backend_type));
                self.config.metrics.storage.backend
            }
        };

        self
    }

    // === Event Configuration ===

    /// Enable or disable event tracking
    pub fn enable_events(mut self, enabled: bool) -> Self {
        self.config.events.enabled = enabled;
        self
    }

    /// Set event buffer size
    pub fn event_buffer_size(mut self, size: usize) -> Self {
        if size == 0 {
            self.validation_errors.push("Event buffer size must be positive".to_string());
        }
        self.config.events.collection.buffer_size = size;
        self
    }

    /// Configure event collection method
    pub fn event_collection_method(mut self, method: &str, endpoint: &str) -> Self {
        use super::event_tracking::CollectionMethod;

        self.config.events.collection.method = match method {
            "push" => CollectionMethod::Push {
                endpoint: endpoint.to_string(),
                max_connections: 10,
            },
            "file" => CollectionMethod::File {
                watch_directory: endpoint.to_string(),
                file_pattern: "*.log".to_string(),
            },
            _ => {
                self.validation_errors.push(format!("Unknown collection method: {}", method));
                self.config.events.collection.method
            }
        };

        self
    }

    // === Performance Configuration ===

    /// Enable or disable performance monitoring
    pub fn enable_performance_monitoring(mut self, enabled: bool) -> Self {
        self.config.performance.enabled = enabled;
        self
    }

    /// Enable performance profiling
    pub fn performance_profiling(mut self, enabled: bool) -> Self {
        self.config.performance.profiling.enabled = enabled;
        self
    }

    /// Set profiling sampling rate
    pub fn profiling_sampling_rate(mut self, rate: f64) -> Self {
        if rate < 0.0 || rate > 1.0 {
            self.validation_errors.push("Profiling sampling rate must be between 0.0 and 1.0".to_string());
        }
        self.config.performance.profiling.sampling_rate = rate;
        self
    }

    /// Configure benchmarking
    pub fn enable_benchmarking(mut self, enabled: bool) -> Self {
        self.config.performance.benchmarking.enabled = enabled;
        self
    }

    // === Resource Monitoring Configuration ===

    /// Enable resource monitoring for all resource types
    pub fn resource_monitoring_all(mut self) -> Self {
        use super::resource_monitoring::ResourceType;

        self.config.resources.enabled = true;
        self.config.resources.resource_types = vec![
            ResourceType::Cpu { per_core: true, frequency_scaling: true, temperature: true },
            ResourceType::Memory { allocation_tracking: true, swap_monitoring: true, fragmentation_analysis: true },
            ResourceType::Storage { all_filesystems: true, io_monitoring: true, health_monitoring: true },
            ResourceType::Network { all_interfaces: true, connection_tracking: true, latency_monitoring: true },
            ResourceType::Process { all_processes: false, file_descriptors: true, thread_monitoring: true },
        ];
        self
    }

    /// Set CPU utilization thresholds
    pub fn cpu_thresholds(mut self, warning: f64, critical: f64) -> Self {
        if warning >= critical {
            self.validation_errors.push("Warning threshold must be less than critical threshold".to_string());
        }
        self.config.resources.thresholds.cpu.utilization_warning = warning;
        self.config.resources.thresholds.cpu.utilization_critical = critical;
        self
    }

    /// Set memory usage thresholds
    pub fn memory_thresholds(mut self, warning: f64, critical: f64) -> Self {
        if warning >= critical {
            self.validation_errors.push("Memory warning threshold must be less than critical threshold".to_string());
        }
        self.config.resources.thresholds.memory.usage_warning = warning;
        self.config.resources.thresholds.memory.usage_critical = critical;
        self
    }

    // === Alert Configuration ===

    /// Enable or disable alerting
    pub fn enable_alerts(mut self, enabled: bool) -> Self {
        self.config.alerts.enabled = enabled;
        self
    }

    /// Add alert channel
    pub fn alert_channel(mut self, channel_type: &str) -> Self {
        use super::alert_configuration::{AlertChannel, SlackFormatting};

        let channel = match channel_type {
            "slack" => AlertChannel::Slack {
                name: "default_slack".to_string(),
                webhook_url: "CONFIGURE_WEBHOOK_URL".to_string(),
                channel: "#alerts".to_string(),
                formatting: SlackFormatting {
                    rich_formatting: true,
                    include_context: true,
                    colors: HashMap::new(),
                },
            },
            "email" => {
                use super::alert_configuration::{SmtpConfig, EmailTemplate};
                AlertChannel::Email {
                    name: "default_email".to_string(),
                    smtp_config: SmtpConfig {
                        host: "CONFIGURE_SMTP_HOST".to_string(),
                        port: 587,
                        use_tls: true,
                        auth: None,
                        from: "alerts@company.com".to_string(),
                    },
                    recipients: vec!["admin@company.com".to_string()],
                    template: EmailTemplate {
                        subject: "[{{severity}}] {{alert_name}}".to_string(),
                        body_html: "<h2>{{alert_name}}</h2><p>{{description}}</p>".to_string(),
                        body_text: "{{alert_name}}: {{description}}".to_string(),
                        include_details: true,
                    },
                }
            },
            "pagerduty" => {
                use super::alert_configuration::{PagerDutyService};
                AlertChannel::PagerDuty {
                    name: "default_pagerduty".to_string(),
                    integration_key: "CONFIGURE_PAGERDUTY_KEY".to_string(),
                    service: PagerDutyService {
                        name: "Monitoring Service".to_string(),
                        id: "SERVICE_ID".to_string(),
                        escalation_policy: "default".to_string(),
                    },
                }
            },
            _ => {
                self.validation_errors.push(format!("Unknown alert channel type: {}", channel_type));
                return self;
            }
        };

        self.config.alerts.channels.push(channel);
        self
    }

    /// Add threshold-based alert rule
    pub fn alert_threshold(mut self, name: &str, metric: &str, threshold: f64) -> Self {
        use super::alert_configuration::AlertRule;

        let rule = AlertRule::threshold(name, metric, threshold);
        self.config.alerts.rules.push(rule);
        self
    }

    // === Data Management Configuration ===

    /// Set sampling rate
    pub fn sampling_rate(mut self, rate: f64) -> Self {
        if rate < 0.0 || rate > 1.0 {
            self.validation_errors.push("Sampling rate must be between 0.0 and 1.0".to_string());
        }
        self.config.sampling.rate = rate;
        self.config.sampling.enabled = rate < 1.0;
        self
    }

    /// Enable or disable data export
    pub fn enable_export(mut self, enabled: bool) -> Self {
        self.config.export.enabled = enabled;
        self
    }

    /// Set export format
    pub fn export_format(mut self, format: &str) -> Self {
        use super::data_management::ExportFormat;

        let export_format = match format {
            "json" => ExportFormat::Json { pretty_print: true, compression: None },
            "csv" => ExportFormat::Csv { delimiter: ',', include_headers: true, quote_char: '"' },
            "parquet" => ExportFormat::Parquet { compression: "snappy".to_string(), row_group_size: 1000000 },
            _ => {
                self.validation_errors.push(format!("Unknown export format: {}", format));
                return self;
            }
        };

        self.config.export.formats = vec![export_format];
        self
    }

    /// Set data retention period
    pub fn retention_period(mut self, period: Duration) -> Self {
        self.config.retention.metrics_retention = period;
        self.config.retention.events_retention = period;
        self
    }

    /// Configure archiving
    pub fn enable_archiving(mut self, enabled: bool) -> Self {
        self.config.retention.archive.enabled = enabled;
        self
    }

    // === Health Check Configuration ===

    /// Configure health checks
    pub fn health_checks(mut self, enabled: bool, interval: Duration) -> Self {
        self.config.health_checks.enabled = enabled;
        self.config.health_checks.interval = interval;

        if interval < Duration::from_secs(1) {
            self.validation_errors.push("Health check interval must be at least 1 second".to_string());
        }

        self
    }

    /// Add health check endpoint
    pub fn health_check_endpoint(mut self, name: &str, endpoint: &str) -> Self {
        let health_endpoint = HealthCheckEndpoint {
            name: name.to_string(),
            endpoint: endpoint.to_string(),
            expected_response: "OK".to_string(),
            timeout: Duration::from_secs(5),
        };

        self.config.health_checks.endpoints.push(health_endpoint);
        self
    }

    // === Advanced Configuration ===

    /// Apply custom configuration function
    pub fn configure<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut MonitoringConfig),
    {
        f(&mut self.config);
        self
    }

    /// Load configuration from environment variables
    pub fn from_environment(mut self) -> Self {
        // Load common environment variables
        if let Ok(val) = std::env::var("MONITORING_METRICS_ENABLED") {
            if let Ok(enabled) = val.parse::<bool>() {
                self.config.metrics.enabled = enabled;
            }
        }

        if let Ok(val) = std::env::var("MONITORING_SAMPLING_RATE") {
            if let Ok(rate) = val.parse::<f64>() {
                if rate >= 0.0 && rate <= 1.0 {
                    self.config.sampling.rate = rate;
                    self.config.sampling.enabled = rate < 1.0;
                }
            }
        }

        if let Ok(val) = std::env::var("MONITORING_ALERTS_ENABLED") {
            if let Ok(enabled) = val.parse::<bool>() {
                self.config.alerts.enabled = enabled;
            }
        }

        // Add more environment variable loading as needed
        self
    }

    /// Load configuration from file
    pub fn from_file(mut self, file_path: &str) -> Self {
        // This would load configuration from a JSON/YAML file
        // Implementation would depend on the chosen serialization format
        self.validation_errors.push(format!("Configuration file loading not yet implemented: {}", file_path));
        self
    }

    /// Validate configuration and build final config
    pub fn build(mut self) -> SklResult<MonitoringConfig> {
        // Perform final validation
        self.validate_configuration()?;

        // Check for any validation errors collected during building
        if !self.validation_errors.is_empty() {
            return Err(SklearsError::InvalidParameter(
                format!("Configuration validation failed: {}", self.validation_errors.join("; "))
            ));
        }

        // Perform cross-component validation
        self.config.validate()?;

        Ok(self.config)
    }

    /// Validate the current configuration state
    fn validate_configuration(&mut self) -> SklResult<()> {
        // Validate metrics configuration
        if let Err(e) = self.config.metrics.validate() {
            self.validation_errors.push(format!("Metrics validation failed: {}", e));
        }

        // Validate events configuration
        if let Err(e) = self.config.events.validate() {
            self.validation_errors.push(format!("Events validation failed: {}", e));
        }

        // Validate performance configuration
        if let Err(e) = self.config.performance.validate() {
            self.validation_errors.push(format!("Performance validation failed: {}", e));
        }

        // Validate resource monitoring configuration
        if let Err(e) = self.config.resources.validate() {
            self.validation_errors.push(format!("Resources validation failed: {}", e));
        }

        // Validate alert configuration
        if let Err(e) = self.config.alerts.validate() {
            self.validation_errors.push(format!("Alerts validation failed: {}", e));
        }

        // Environment-specific validation
        self.validate_environment_specific()?;

        Ok(())
    }

    /// Validate environment-specific constraints
    fn validate_environment_specific(&mut self) -> SklResult<()> {
        match self.environment {
            EnvironmentProfile::Production | EnvironmentProfile::HighVolume => {
                // Production validation
                if !self.config.alerts.enabled {
                    self.validation_errors.push("Alerts should be enabled in production".to_string());
                }

                if self.config.sampling.rate > 0.1 && matches!(self.environment, EnvironmentProfile::HighVolume) {
                    self.validation_errors.push("High volume environments should use lower sampling rates".to_string());
                }
            },
            EnvironmentProfile::SecurityFocused => {
                // Security validation
                if self.config.export.enabled {
                    self.validation_errors.push("Data export should be carefully considered in security-focused environments".to_string());
                }

                if !self.config.retention.archive.encryption.enabled {
                    self.validation_errors.push("Encryption should be enabled for security-focused environments".to_string());
                }
            },
            EnvironmentProfile::ComplianceFocused => {
                // Compliance validation
                if self.config.sampling.enabled {
                    self.validation_errors.push("Sampling may not be appropriate for compliance environments".to_string());
                }

                if !self.config.retention.archive.enabled {
                    self.validation_errors.push("Archiving should be enabled for compliance environments".to_string());
                }
            },
            _ => {
                // Other environments have fewer constraints
            }
        }

        Ok(())
    }

    /// Get configuration summary for review
    pub fn summary(&self) -> ConfigurationSummary {
        ConfigurationSummary {
            environment: self.environment.clone(),
            metrics_enabled: self.config.metrics.enabled,
            events_enabled: self.config.events.enabled,
            performance_enabled: self.config.performance.enabled,
            alerts_enabled: self.config.alerts.enabled,
            sampling_enabled: self.config.sampling.enabled,
            sampling_rate: self.config.sampling.rate,
            export_enabled: self.config.export.enabled,
            validation_errors: self.validation_errors.clone(),
        }
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration summary for review and validation
#[derive(Debug, Clone)]
pub struct ConfigurationSummary {
    /// Environment profile being used
    pub environment: EnvironmentProfile,
    /// Whether metrics are enabled
    pub metrics_enabled: bool,
    /// Whether events are enabled
    pub events_enabled: bool,
    /// Whether performance monitoring is enabled
    pub performance_enabled: bool,
    /// Whether alerts are enabled
    pub alerts_enabled: bool,
    /// Whether sampling is enabled
    pub sampling_enabled: bool,
    /// Current sampling rate
    pub sampling_rate: f64,
    /// Whether export is enabled
    pub export_enabled: bool,
    /// Validation errors found during building
    pub validation_errors: Vec<String>,
}

/// Health check configuration
///
/// Defines health monitoring endpoints and criteria for determining
/// system health status. Health checks provide automated system
/// health visibility and can trigger alerts or automated responses.
///
/// # Health Check Types
///
/// Different types of health checks serve different purposes:
/// - **Endpoint checks**: HTTP/TCP endpoint availability
/// - **Service checks**: Application-specific health indicators
/// - **Resource checks**: System resource availability
/// - **Dependency checks**: External service availability
///
/// # Usage Examples
///
/// ## Basic Health Check Configuration
/// ```rust
/// use sklears_compose::monitoring_config::HealthCheckConfig;
///
/// let config = HealthCheckConfig::default();
/// ```
///
/// ## Production Health Checks
/// ```rust
/// let config = HealthCheckConfig::production();
/// ```
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,

    /// Health check interval
    ///
    /// How often to perform health checks.
    pub interval: Duration,

    /// Health check timeout
    ///
    /// Maximum time to wait for health check response.
    pub timeout: Duration,

    /// Health check endpoints
    ///
    /// List of endpoints to monitor for health status.
    pub endpoints: Vec<HealthCheckEndpoint>,

    /// Health criteria
    ///
    /// Criteria for determining overall system health.
    pub criteria: HealthCriteria,

    /// Health check retries
    ///
    /// Retry configuration for failed health checks.
    pub retries: HealthCheckRetries,

    /// Health check notifications
    ///
    /// Notification settings for health status changes.
    pub notifications: HealthCheckNotifications,
}

/// Health check endpoint definition
#[derive(Debug, Clone)]
pub struct HealthCheckEndpoint {
    /// Endpoint name for identification
    pub name: String,

    /// Endpoint URL or path
    ///
    /// Can be HTTP URL, TCP address, or file path depending on check type.
    pub endpoint: String,

    /// Expected response for success
    ///
    /// What response indicates a healthy endpoint.
    pub expected_response: String,

    /// Timeout for this specific endpoint
    ///
    /// Override the global timeout for this endpoint.
    pub timeout: Duration,

    /// Health check type
    pub check_type: HealthCheckType,

    /// Additional headers (for HTTP checks)
    pub headers: HashMap<String, String>,

    /// Authentication (for HTTP checks)
    pub auth: Option<HealthCheckAuth>,
}

/// Types of health checks
#[derive(Debug, Clone)]
pub enum HealthCheckType {
    /// HTTP/HTTPS endpoint check
    Http {
        /// HTTP method to use
        method: String,
        /// Expected HTTP status code
        expected_status: u16,
    },

    /// TCP port check
    Tcp {
        port: u16,
    },

    /// Command execution check
    Command {
        /// Command to execute
        command: String,
        /// Expected exit code
        expected_exit_code: i32,
    },

    /// File existence check
    File {
        /// File path to check
        path: String,
    },

    /// Custom health check
    Custom {
        /// Check type identifier
        check_type: String,
        /// Check parameters
        parameters: HashMap<String, String>,
    },
}

/// Authentication for health check endpoints
#[derive(Debug, Clone)]
pub enum HealthCheckAuth {
    /// Basic authentication
    Basic { username: String, password: String },
    /// Bearer token authentication
    Bearer { token: String },
    /// API key authentication
    ApiKey { key: String, header: String },
}

/// Health criteria for overall system health
#[derive(Debug, Clone)]
pub struct HealthCriteria {
    /// Response time threshold for healthy status
    pub response_time_threshold: Duration,

    /// Error rate threshold for healthy status
    pub error_rate_threshold: f64,

    /// Availability threshold for healthy status
    pub availability_threshold: f64,

    /// Custom health metrics and thresholds
    pub custom_metrics: Vec<CustomHealthMetric>,

    /// Minimum number of healthy endpoints
    pub min_healthy_endpoints: usize,

    /// Health aggregation strategy
    pub aggregation_strategy: HealthAggregationStrategy,
}

/// Custom health metric definition
#[derive(Debug, Clone)]
pub struct CustomHealthMetric {
    /// Metric name
    pub name: String,
    /// Healthy threshold
    pub healthy_threshold: f64,
    /// Comparison operator
    pub operator: HealthComparisonOperator,
    /// Weight in overall health calculation
    pub weight: f64,
}

/// Comparison operators for health thresholds
#[derive(Debug, Clone)]
pub enum HealthComparisonOperator {
    /// Greater than threshold
    GreaterThan,
    /// Less than threshold
    LessThan,
    /// Equal to threshold
    Equal,
    /// Not equal to threshold
    NotEqual,
}

/// Health aggregation strategies
#[derive(Debug, Clone)]
pub enum HealthAggregationStrategy {
    /// All endpoints must be healthy
    All,
    /// Majority of endpoints must be healthy
    Majority,
    /// At least N endpoints must be healthy
    AtLeast(usize),
    /// Weighted average based on endpoint importance
    Weighted,
}

/// Health check retry configuration
#[derive(Debug, Clone)]
pub struct HealthCheckRetries {
    /// Enable retries for failed health checks
    pub enabled: bool,

    /// Maximum number of retries
    pub max_retries: u32,

    /// Delay between retries
    pub retry_delay: Duration,

    /// Exponential backoff for retries
    pub exponential_backoff: bool,

    /// Maximum total retry time
    pub max_retry_time: Duration,
}

/// Health check notification configuration
#[derive(Debug, Clone)]
pub struct HealthCheckNotifications {
    /// Enable health status notifications
    pub enabled: bool,

    /// Notification channels for health status changes
    pub channels: Vec<String>,

    /// Notify on health status changes
    pub notify_on_status_change: bool,

    /// Notify on health degradation
    pub notify_on_degradation: bool,

    /// Notify on health recovery
    pub notify_on_recovery: bool,

    /// Notification throttling
    pub throttling: NotificationThrottling,
}

/// Notification throttling configuration
#[derive(Debug, Clone)]
pub struct NotificationThrottling {
    /// Enable notification throttling
    pub enabled: bool,

    /// Minimum time between notifications
    pub min_interval: Duration,

    /// Maximum notifications per time window
    pub max_notifications: u32,

    /// Time window for notification counting
    pub time_window: Duration,
}

impl HealthCheckConfig {
    /// Create configuration optimized for production environments
    pub fn production() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            endpoints: vec![
                HealthCheckEndpoint {
                    name: "api_health".to_string(),
                    endpoint: "http://localhost:8080/health".to_string(),
                    expected_response: "OK".to_string(),
                    timeout: Duration::from_secs(5),
                    check_type: HealthCheckType::Http {
                        method: "GET".to_string(),
                        expected_status: 200,
                    },
                    headers: HashMap::new(),
                    auth: None,
                },
                HealthCheckEndpoint {
                    name: "database_health".to_string(),
                    endpoint: "localhost:5432".to_string(),
                    expected_response: "connected".to_string(),
                    timeout: Duration::from_secs(3),
                    check_type: HealthCheckType::Tcp { port: 5432 },
                    headers: HashMap::new(),
                    auth: None,
                },
            ],
            criteria: HealthCriteria {
                response_time_threshold: Duration::from_millis(1000),
                error_rate_threshold: 0.05,
                availability_threshold: 0.99,
                custom_metrics: Vec::new(),
                min_healthy_endpoints: 1,
                aggregation_strategy: HealthAggregationStrategy::Majority,
            },
            retries: HealthCheckRetries {
                enabled: true,
                max_retries: 3,
                retry_delay: Duration::from_secs(1),
                exponential_backoff: true,
                max_retry_time: Duration::from_secs(30),
            },
            notifications: HealthCheckNotifications {
                enabled: true,
                channels: vec!["health_alerts".to_string()],
                notify_on_status_change: true,
                notify_on_degradation: true,
                notify_on_recovery: true,
                throttling: NotificationThrottling {
                    enabled: true,
                    min_interval: Duration::from_secs(300),
                    max_notifications: 5,
                    time_window: Duration::from_secs(3600),
                },
            },
        }
    }

    /// Create configuration optimized for development environments
    pub fn development() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            timeout: Duration::from_secs(5),
            endpoints: vec![
                HealthCheckEndpoint {
                    name: "local_api".to_string(),
                    endpoint: "http://localhost:3000/health".to_string(),
                    expected_response: "OK".to_string(),
                    timeout: Duration::from_secs(5),
                    check_type: HealthCheckType::Http {
                        method: "GET".to_string(),
                        expected_status: 200,
                    },
                    headers: HashMap::new(),
                    auth: None,
                },
            ],
            criteria: HealthCriteria {
                response_time_threshold: Duration::from_millis(5000),
                error_rate_threshold: 0.1,
                availability_threshold: 0.95,
                custom_metrics: Vec::new(),
                min_healthy_endpoints: 0,
                aggregation_strategy: HealthAggregationStrategy::AtLeast(0),
            },
            retries: HealthCheckRetries {
                enabled: false,
                max_retries: 1,
                retry_delay: Duration::from_secs(1),
                exponential_backoff: false,
                max_retry_time: Duration::from_secs(10),
            },
            notifications: HealthCheckNotifications {
                enabled: false,
                channels: Vec::new(),
                notify_on_status_change: false,
                notify_on_degradation: false,
                notify_on_recovery: false,
                throttling: NotificationThrottling {
                    enabled: false,
                    min_interval: Duration::from_secs(600),
                    max_notifications: 3,
                    time_window: Duration::from_secs(1800),
                },
            },
        }
    }

    /// Validate health check configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate intervals
        if self.timeout >= self.interval {
            return Err("Health check timeout must be less than interval".to_string());
        }

        // Validate endpoints
        for endpoint in &self.endpoints {
            if endpoint.name.is_empty() {
                return Err("Health check endpoint name cannot be empty".to_string());
            }

            if endpoint.endpoint.is_empty() {
                return Err("Health check endpoint URL cannot be empty".to_string());
            }
        }

        // Validate criteria
        if self.criteria.error_rate_threshold < 0.0 || self.criteria.error_rate_threshold > 1.0 {
            return Err("Error rate threshold must be between 0.0 and 1.0".to_string());
        }

        if self.criteria.availability_threshold < 0.0 || self.criteria.availability_threshold > 1.0 {
            return Err("Availability threshold must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            endpoints: Vec::new(),
            criteria: HealthCriteria {
                response_time_threshold: Duration::from_millis(1000),
                error_rate_threshold: 0.05,
                availability_threshold: 0.99,
                custom_metrics: Vec::new(),
                min_healthy_endpoints: 0,
                aggregation_strategy: HealthAggregationStrategy::Majority,
            },
            retries: HealthCheckRetries {
                enabled: true,
                max_retries: 3,
                retry_delay: Duration::from_secs(1),
                exponential_backoff: false,
                max_retry_time: Duration::from_secs(30),
            },
            notifications: HealthCheckNotifications {
                enabled: false,
                channels: Vec::new(),
                notify_on_status_change: false,
                notify_on_degradation: true,
                notify_on_recovery: true,
                throttling: NotificationThrottling {
                    enabled: true,
                    min_interval: Duration::from_secs(300),
                    max_notifications: 10,
                    time_window: Duration::from_secs(3600),
                },
            },
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder_basic() {
        let config = ConfigBuilder::new()
            .enable_metrics(true)
            .enable_alerts(false)
            .sampling_rate(0.5)
            .build()
            .unwrap();

        assert!(config.metrics.enabled);
        assert!(!config.alerts.enabled);
        assert_eq!(config.sampling.rate, 0.5);
    }

    #[test]
    fn test_config_builder_production() {
        let config = ConfigBuilder::production()
            .build()
            .unwrap();

        assert!(config.metrics.enabled);
        assert!(config.alerts.enabled);
        assert!(config.health_checks.enabled);
    }

    #[test]
    fn test_config_builder_validation() {
        let result = ConfigBuilder::new()
            .sampling_rate(1.5) // Invalid rate
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_health_check_config_validation() {
        let config = HealthCheckConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = HealthCheckConfig::default();
        invalid_config.criteria.error_rate_threshold = 1.5; // Invalid threshold
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_environment_profiles() {
        let prod_config = ConfigBuilder::production().build().unwrap();
        let dev_config = ConfigBuilder::development().build().unwrap();

        // Production should have more features enabled
        assert!(prod_config.alerts.enabled);
        assert!(prod_config.performance.enabled);

        // Development should be more relaxed
        assert!(!dev_config.alerts.enabled);
    }
}