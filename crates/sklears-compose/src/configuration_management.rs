//! Configuration Management for Execution Monitoring
//!
//! This module provides comprehensive configuration management capabilities for the execution
//! monitoring framework, including configuration loading, validation, merging, templating,
//! change notification, security, and versioning.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime};
use std::fs;
use std::fmt;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde_json::{Value, Map};
use chrono::{DateTime, Utc};

/// Comprehensive configuration structure for execution monitoring
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MonitoringConfiguration {
    /// Configuration metadata
    pub metadata: ConfigurationMetadata,
    /// Core monitoring settings
    pub monitoring: MonitoringSettings,
    /// Metrics collection configuration
    pub metrics: MetricsConfiguration,
    /// Event tracking configuration
    pub events: EventConfiguration,
    /// Alerting system configuration
    pub alerts: AlertConfiguration,
    /// Health monitoring configuration
    pub health: HealthConfiguration,
    /// Performance analysis configuration
    pub performance: PerformanceConfiguration,
    /// Reporting configuration
    pub reporting: ReportingConfiguration,
    /// Data retention configuration
    pub retention: RetentionConfiguration,
    /// Security and encryption settings
    pub security: SecurityConfiguration,
    /// Feature flags and toggles
    pub features: FeatureConfiguration,
    /// Environment-specific overrides
    pub environment: EnvironmentConfiguration,
    /// Plugin and extension configuration
    pub plugins: PluginConfiguration,
}

/// Configuration metadata and versioning information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigurationMetadata {
    /// Configuration schema version
    pub version: String,
    /// Configuration creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modification timestamp
    pub modified_at: DateTime<Utc>,
    /// Configuration checksum for integrity verification
    pub checksum: String,
    /// Configuration source (file, environment, default, etc.)
    pub source: ConfigurationSource,
    /// Configuration tags for categorization
    pub tags: Vec<String>,
    /// Configuration description
    pub description: Option<String>,
    /// Configuration author/creator
    pub author: Option<String>,
}

/// Configuration source enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConfigurationSource {
    /// Loaded from configuration file
    File(PathBuf),
    /// Loaded from environment variables
    Environment,
    /// Default/built-in configuration
    Default,
    /// Merged from multiple sources
    Merged(Vec<ConfigurationSource>),
    /// Dynamic/runtime configuration
    Dynamic,
    /// Remote configuration service
    Remote(String),
}

/// Core monitoring framework settings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MonitoringSettings {
    /// Enable/disable monitoring globally
    pub enabled: bool,
    /// Monitoring sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Maximum concurrent monitoring sessions
    pub max_sessions: usize,
    /// Session timeout duration
    pub session_timeout: Duration,
    /// Monitoring thread pool size
    pub thread_pool_size: usize,
    /// Buffer sizes for various components
    pub buffer_sizes: BufferSizes,
    /// Logging configuration
    pub logging: LoggingConfiguration,
}

/// Buffer size configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BufferSizes {
    /// Event buffer size
    pub events: usize,
    /// Metrics buffer size
    pub metrics: usize,
    /// Alert buffer size
    pub alerts: usize,
    /// Health check buffer size
    pub health_checks: usize,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoggingConfiguration {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Log output format (json, plain, structured)
    pub format: String,
    /// Log rotation settings
    pub rotation: LogRotationSettings,
    /// Log filters and exclusions
    pub filters: Vec<String>,
}

/// Log rotation settings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogRotationSettings {
    /// Maximum log file size before rotation
    pub max_size: u64,
    /// Maximum number of rotated log files to keep
    pub max_files: usize,
    /// Log compression enabled
    pub compress: bool,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricsConfiguration {
    /// Enable metrics collection
    pub enabled: bool,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Metrics aggregation window
    pub aggregation_window: Duration,
    /// Maximum number of metrics to retain in memory
    pub max_memory_metrics: usize,
    /// Metrics persistence configuration
    pub persistence: MetricsPersistenceConfig,
    /// Custom metrics definitions
    pub custom_metrics: Vec<CustomMetricDefinition>,
}

/// Metrics persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricsPersistenceConfig {
    /// Enable metrics persistence
    pub enabled: bool,
    /// Persistence backend type
    pub backend: String,
    /// Backend-specific configuration
    pub backend_config: Map<String, Value>,
    /// Persistence interval
    pub persistence_interval: Duration,
}

/// Custom metric definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomMetricDefinition {
    /// Metric name
    pub name: String,
    /// Metric type (counter, gauge, histogram, timer)
    pub metric_type: String,
    /// Metric description
    pub description: String,
    /// Metric labels/tags
    pub labels: Vec<String>,
    /// Metric collection configuration
    pub collection: CustomMetricCollection,
}

/// Custom metric collection configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomMetricCollection {
    /// Collection enabled
    pub enabled: bool,
    /// Collection interval
    pub interval: Duration,
    /// Collection source/method
    pub source: String,
    /// Collection parameters
    pub parameters: Map<String, Value>,
}

/// Event tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventConfiguration {
    /// Enable event tracking
    pub enabled: bool,
    /// Event buffer size
    pub buffer_size: usize,
    /// Event processing batch size
    pub batch_size: usize,
    /// Event processing interval
    pub processing_interval: Duration,
    /// Event filtering rules
    pub filters: Vec<EventFilterRule>,
    /// Event enrichment configuration
    pub enrichment: EventEnrichmentConfig,
}

/// Event filter rule
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventFilterRule {
    /// Rule name
    pub name: String,
    /// Rule enabled
    pub enabled: bool,
    /// Filter condition
    pub condition: String,
    /// Filter action (include, exclude, transform)
    pub action: String,
    /// Filter parameters
    pub parameters: Map<String, Value>,
}

/// Event enrichment configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventEnrichmentConfig {
    /// Enrichment enabled
    pub enabled: bool,
    /// Enrichment sources
    pub sources: Vec<String>,
    /// Enrichment rules
    pub rules: Vec<EnrichmentRule>,
}

/// Event enrichment rule
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnrichmentRule {
    /// Rule name
    pub name: String,
    /// Source field
    pub source_field: String,
    /// Target field
    pub target_field: String,
    /// Enrichment method
    pub method: String,
    /// Method parameters
    pub parameters: Map<String, Value>,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlertConfiguration {
    /// Enable alerting
    pub enabled: bool,
    /// Alert evaluation interval
    pub evaluation_interval: Duration,
    /// Alert rule definitions
    pub rules: Vec<AlertRuleDefinition>,
    /// Alert channels configuration
    pub channels: Vec<AlertChannelDefinition>,
    /// Alert suppression settings
    pub suppression: AlertSuppressionConfig,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlertRuleDefinition {
    /// Rule name
    pub name: String,
    /// Rule enabled
    pub enabled: bool,
    /// Rule condition expression
    pub condition: String,
    /// Alert severity
    pub severity: String,
    /// Alert message template
    pub message_template: String,
    /// Rule evaluation frequency
    pub evaluation_frequency: Duration,
    /// Alert cooldown period
    pub cooldown_period: Duration,
    /// Target channels
    pub channels: Vec<String>,
}

/// Alert channel definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlertChannelDefinition {
    /// Channel name
    pub name: String,
    /// Channel type (email, slack, webhook, etc.)
    pub channel_type: String,
    /// Channel enabled
    pub enabled: bool,
    /// Channel configuration
    pub config: Map<String, Value>,
    /// Channel rate limiting
    pub rate_limit: Option<RateLimitConfig>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RateLimitConfig {
    /// Maximum number of alerts per time window
    pub max_alerts: usize,
    /// Time window duration
    pub time_window: Duration,
}

/// Alert suppression configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlertSuppressionConfig {
    /// Suppression enabled
    pub enabled: bool,
    /// Suppression rules
    pub rules: Vec<SuppressionRule>,
    /// Default suppression duration
    pub default_duration: Duration,
}

/// Alert suppression rule
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SuppressionRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Suppression duration
    pub duration: Duration,
    /// Rule priority
    pub priority: i32,
}

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthConfiguration {
    /// Enable health monitoring
    pub enabled: bool,
    /// Health check interval
    pub check_interval: Duration,
    /// Health check timeout
    pub check_timeout: Duration,
    /// Health check definitions
    pub checks: Vec<HealthCheckDefinition>,
    /// Health thresholds
    pub thresholds: HealthThresholds,
}

/// Health check definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthCheckDefinition {
    /// Check name
    pub name: String,
    /// Check enabled
    pub enabled: bool,
    /// Check type
    pub check_type: String,
    /// Check configuration
    pub config: Map<String, Value>,
    /// Check interval override
    pub interval: Option<Duration>,
    /// Check timeout override
    pub timeout: Option<Duration>,
}

/// Health monitoring thresholds
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthThresholds {
    /// CPU usage threshold (percentage)
    pub cpu_usage: f64,
    /// Memory usage threshold (percentage)
    pub memory_usage: f64,
    /// Disk usage threshold (percentage)
    pub disk_usage: f64,
    /// Network latency threshold (milliseconds)
    pub network_latency: Duration,
    /// Error rate threshold (percentage)
    pub error_rate: f64,
}

/// Performance analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerformanceConfiguration {
    /// Enable performance analysis
    pub enabled: bool,
    /// Analysis interval
    pub analysis_interval: Duration,
    /// Analysis window size
    pub window_size: Duration,
    /// Anomaly detection configuration
    pub anomaly_detection: AnomalyDetectionConfig,
    /// Trend analysis configuration
    pub trend_analysis: TrendAnalysisConfig,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnomalyDetectionConfig {
    /// Anomaly detection enabled
    pub enabled: bool,
    /// Detection algorithms to use
    pub algorithms: Vec<String>,
    /// Sensitivity level (0.0 to 1.0)
    pub sensitivity: f64,
    /// Minimum data points required
    pub min_data_points: usize,
}

/// Trend analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrendAnalysisConfig {
    pub enabled: bool,
    pub methods: Vec<String>,
    pub significance_threshold: f64,
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReportingConfiguration {
    /// Enable reporting
    pub enabled: bool,
    /// Report generation interval
    pub generation_interval: Duration,
    /// Report templates
    pub templates: Vec<ReportTemplate>,
    /// Report distribution settings
    pub distribution: ReportDistributionConfig,
}

/// Report template definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,
    /// Template enabled
    pub enabled: bool,
    /// Template format (html, pdf, json, csv)
    pub format: String,
    /// Template content/configuration
    pub content: Map<String, Value>,
    /// Generation schedule
    pub schedule: String,
}

/// Report distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReportDistributionConfig {
    /// Distribution enabled
    pub enabled: bool,
    /// Distribution channels
    pub channels: Vec<String>,
    /// Distribution schedule
    pub schedule: String,
    /// Distribution filters
    pub filters: Vec<String>,
}

/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetentionConfiguration {
    /// Enable data retention policies
    pub enabled: bool,
    /// Default retention period
    pub default_retention: Duration,
    /// Data type specific retention policies
    pub policies: Vec<RetentionPolicy>,
    /// Cleanup schedule
    pub cleanup_schedule: String,
}

/// Data retention policy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetentionPolicy {
    /// Policy name
    pub name: String,
    /// Data type pattern
    pub data_type: String,
    /// Retention duration
    pub retention_duration: Duration,
    /// Archival settings
    pub archival: Option<ArchivalConfig>,
}

/// Data archival configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArchivalConfig {
    /// Archival enabled
    pub enabled: bool,
    /// Archival backend
    pub backend: String,
    /// Archival configuration
    pub config: Map<String, Value>,
}

/// Security and encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecurityConfiguration {
    /// Enable security features
    pub enabled: bool,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
    /// Authorization configuration
    pub authorization: AuthorizationConfig,
    /// Audit logging configuration
    pub audit: AuditConfig,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EncryptionConfig {
    /// Encryption enabled
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: String,
    /// Key management configuration
    pub key_management: KeyManagementConfig,
    /// Data encryption scope
    pub scope: Vec<String>,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeyManagementConfig {
    /// Key provider type
    pub provider: String,
    /// Key rotation enabled
    pub rotation_enabled: bool,
    /// Key rotation interval
    pub rotation_interval: Duration,
    /// Key configuration
    pub config: Map<String, Value>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuthenticationConfig {
    /// Authentication enabled
    pub enabled: bool,
    /// Authentication methods
    pub methods: Vec<String>,
    /// Token configuration
    pub token: TokenConfig,
}

/// Token configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenConfig {
    /// Token type
    pub token_type: String,
    /// Token expiration
    pub expiration: Duration,
    /// Token refresh enabled
    pub refresh_enabled: bool,
}

/// Authorization configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuthorizationConfig {
    /// Authorization enabled
    pub enabled: bool,
    /// Authorization model (rbac, abac, etc.)
    pub model: String,
    /// Permission definitions
    pub permissions: Vec<PermissionDefinition>,
}

/// Permission definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PermissionDefinition {
    /// Permission name
    pub name: String,
    /// Permission description
    pub description: String,
    /// Required roles
    pub roles: Vec<String>,
    /// Resource patterns
    pub resources: Vec<String>,
    /// Allowed actions
    pub actions: Vec<String>,
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditConfig {
    /// Audit logging enabled
    pub enabled: bool,
    /// Audit events to log
    pub events: Vec<String>,
    /// Audit log format
    pub format: String,
    /// Audit log retention
    pub retention: Duration,
}

/// Feature flags and toggles configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureConfiguration {
    /// Feature flags
    pub flags: HashMap<String, FeatureFlag>,
    /// Feature toggle refresh interval
    pub refresh_interval: Duration,
    /// Remote feature flag configuration
    pub remote: Option<RemoteFeatureConfig>,
}

/// Feature flag definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureFlag {
    /// Flag enabled state
    pub enabled: bool,
    /// Flag description
    pub description: String,
    /// Flag rollout percentage (0.0 to 1.0)
    pub rollout: f64,
    /// Flag conditions
    pub conditions: Vec<FeatureCondition>,
}

/// Feature flag condition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureCondition {
    /// Condition type
    pub condition_type: String,
    /// Condition value
    pub value: Value,
    /// Condition operator
    pub operator: String,
}

/// Remote feature flag configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteFeatureConfig {
    /// Remote service URL
    pub url: String,
    /// API key for authentication
    pub api_key: String,
    /// Polling interval
    pub poll_interval: Duration,
    /// Timeout for remote calls
    pub timeout: Duration,
}

/// Environment-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentConfiguration {
    /// Current environment name
    pub environment: String,
    /// Environment-specific overrides
    pub overrides: Map<String, Value>,
    /// Environment variables mapping
    pub variables: HashMap<String, String>,
    /// Environment validation rules
    pub validation: Vec<EnvironmentValidationRule>,
}

/// Environment validation rule
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentValidationRule {
    /// Rule name
    pub name: String,
    /// Environment pattern
    pub environment_pattern: String,
    /// Required configuration fields
    pub required_fields: Vec<String>,
    /// Validation constraints
    pub constraints: Map<String, Value>,
}

/// Plugin and extension configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PluginConfiguration {
    /// Plugin discovery enabled
    pub discovery_enabled: bool,
    /// Plugin directories
    pub plugin_directories: Vec<PathBuf>,
    /// Loaded plugins
    pub plugins: Vec<PluginDefinition>,
    /// Plugin security settings
    pub security: PluginSecurityConfig,
}

/// Plugin definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PluginDefinition {
    /// Plugin name
    pub name: String,
    /// Plugin enabled
    pub enabled: bool,
    /// Plugin version
    pub version: String,
    /// Plugin path
    pub path: PathBuf,
    /// Plugin configuration
    pub config: Map<String, Value>,
    /// Plugin dependencies
    pub dependencies: Vec<String>,
}

/// Plugin security configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PluginSecurityConfig {
    /// Plugin signature verification enabled
    pub signature_verification: bool,
    /// Allowed plugin sources
    pub allowed_sources: Vec<String>,
    /// Plugin sandboxing enabled
    pub sandboxing: bool,
    /// Resource limits for plugins
    pub resource_limits: PluginResourceLimits,
}

/// Plugin resource limits
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PluginResourceLimits {
    /// Maximum memory usage
    pub max_memory: u64,
    /// Maximum CPU usage percentage
    pub max_cpu: f64,
    /// Maximum file handles
    pub max_file_handles: usize,
    /// Maximum network connections
    pub max_network_connections: usize,
}

/// Configuration manager for loading, validating, and managing configurations
#[derive(Debug)]
pub struct ConfigurationManager {
    /// Current configuration
    current_config: Arc<RwLock<MonitoringConfiguration>>,
    /// Configuration validation schema
    validation_schema: Arc<ConfigurationSchema>,
    /// Configuration change listeners
    listeners: Arc<Mutex<Vec<Box<dyn ConfigurationChangeListener + Send + Sync>>>>,
    /// Configuration cache
    cache: Arc<Mutex<ConfigurationCache>>,
    /// Configuration encryption manager
    encryption: Arc<ConfigurationEncryption>,
    /// Configuration version manager
    versioning: Arc<ConfigurationVersioning>,
}

/// Configuration validation schema
#[derive(Debug)]
pub struct ConfigurationSchema {
    /// Schema version
    pub version: String,
    /// Field validators
    pub validators: HashMap<String, Box<dyn ConfigurationValidator + Send + Sync>>,
    /// Required fields
    pub required_fields: HashSet<String>,
    /// Default values
    pub defaults: Map<String, Value>,
}

/// Configuration change listener trait
pub trait ConfigurationChangeListener: Send + Sync {
    /// Called when configuration changes
    fn on_configuration_changed(&self, old_config: &MonitoringConfiguration, new_config: &MonitoringConfiguration);

    /// Called when configuration validation fails
    fn on_validation_failed(&self, error: &ConfigurationError);
}

/// Configuration validator trait
pub trait ConfigurationValidator: Send + Sync {
    /// Validate a configuration value
    fn validate(&self, value: &Value) -> Result<(), ValidationError>;

    /// Get validation constraints
    fn constraints(&self) -> Vec<ValidationConstraint>;
}

/// Configuration validation constraint
#[derive(Debug, Clone)]
pub struct ValidationConstraint {
    /// Constraint type
    pub constraint_type: String,
    /// Constraint parameters
    pub parameters: Map<String, Value>,
    /// Constraint description
    pub description: String,
}

/// Configuration validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Field path
    pub field: String,
    /// Error message
    pub message: String,
    /// Error code
    pub code: String,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Configuration cache for performance optimization
#[derive(Debug)]
pub struct ConfigurationCache {
    /// Cached configurations by source
    configs: HashMap<String, CachedConfiguration>,
    /// Cache statistics
    stats: CacheStatistics,
    /// Cache settings
    settings: CacheSettings,
}

/// Cached configuration entry
#[derive(Debug, Clone)]
pub struct CachedConfiguration {
    /// Cached configuration
    pub config: MonitoringConfiguration,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Cache TTL
    pub ttl: Duration,
    /// Cache hit count
    pub hit_count: u64,
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total cache evictions
    pub evictions: u64,
    /// Cache memory usage
    pub memory_usage: u64,
}

/// Cache settings
#[derive(Debug)]
pub struct CacheSettings {
    /// Maximum cache size
    pub max_size: usize,
    /// Default TTL
    pub default_ttl: Duration,
    /// Cache enabled
    pub enabled: bool,
}

/// Configuration encryption manager
#[derive(Debug)]
pub struct ConfigurationEncryption {
    /// Encryption algorithm
    algorithm: String,
    /// Encryption key
    key: Vec<u8>,
    /// Encryption enabled
    enabled: bool,
}

/// Configuration versioning manager
#[derive(Debug)]
pub struct ConfigurationVersioning {
    /// Version history
    history: Vec<ConfigurationVersion>,
    /// Current version
    current_version: String,
    /// Versioning enabled
    enabled: bool,
}

/// Configuration version entry
#[derive(Debug, Clone)]
pub struct ConfigurationVersion {
    /// Version identifier
    pub version: String,
    /// Version timestamp
    pub timestamp: SystemTime,
    /// Configuration checksum
    pub checksum: String,
    /// Version description
    pub description: String,
    /// Changes summary
    pub changes: Vec<ConfigurationChange>,
}

/// Configuration change record
#[derive(Debug, Clone)]
pub struct ConfigurationChange {
    /// Change type (added, modified, removed)
    pub change_type: String,
    /// Field path
    pub field: String,
    /// Old value
    pub old_value: Option<Value>,
    /// New value
    pub new_value: Option<Value>,
}

/// Configuration error types
#[derive(Debug, Clone)]
pub enum ConfigurationError {
    /// Configuration file not found
    FileNotFound(PathBuf),
    /// Invalid configuration format
    InvalidFormat(String),
    /// Configuration validation failed
    ValidationFailed(Vec<ValidationError>),
    /// Configuration merge conflict
    MergeConflict(String),
    /// Encryption/decryption error
    EncryptionError(String),
    /// Version compatibility error
    VersionError(String),
    /// Permission denied
    PermissionDenied(String),
    /// Network error for remote configuration
    NetworkError(String),
    /// Generic configuration error
    Other(String),
}

impl fmt::Display for ConfigurationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigurationError::FileNotFound(path) => write!(f, "Configuration file not found: {}", path.display()),
            ConfigurationError::InvalidFormat(msg) => write!(f, "Invalid configuration format: {}", msg),
            ConfigurationError::ValidationFailed(errors) => {
                write!(f, "Configuration validation failed: {}",
                       errors.iter().map(|e| &e.message).collect::<Vec<_>>().join(", "))
            },
            ConfigurationError::MergeConflict(msg) => write!(f, "Configuration merge conflict: {}", msg),
            ConfigurationError::EncryptionError(msg) => write!(f, "Configuration encryption error: {}", msg),
            ConfigurationError::VersionError(msg) => write!(f, "Configuration version error: {}", msg),
            ConfigurationError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            ConfigurationError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ConfigurationError::Other(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for ConfigurationError {}

impl ConfigurationManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        let default_config = Self::default_configuration();
        let schema = Self::default_schema();

        Self {
            current_config: Arc::new(RwLock::new(default_config)),
            validation_schema: Arc::new(schema),
            listeners: Arc::new(Mutex::new(Vec::new())),
            cache: Arc::new(Mutex::new(ConfigurationCache::new())),
            encryption: Arc::new(ConfigurationEncryption::new()),
            versioning: Arc::new(ConfigurationVersioning::new()),
        }
    }

    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(&self, path: P) -> Result<MonitoringConfiguration, ConfigurationError> {
        let path = path.as_ref();

        // Read file contents
        let contents = fs::read_to_string(path)
            .map_err(|_| ConfigurationError::FileNotFound(path.to_path_buf()))?;

        // Parse configuration
        let mut config: MonitoringConfiguration = serde_json::from_str(&contents)
            .map_err(|e| ConfigurationError::InvalidFormat(e.to_string()))?;

        // Set source metadata
        config.metadata.source = ConfigurationSource::File(path.to_path_buf());
        config.metadata.modified_at = Utc::now();

        // Validate configuration
        self.validate_configuration(&config)?;

        // Decrypt if needed
        if self.encryption.enabled {
            config = self.encryption.decrypt_configuration(config)?;
        }

        // Cache configuration
        self.cache_configuration(&config);

        Ok(config)
    }

    /// Load configuration from environment variables
    pub fn load_from_environment(&self) -> Result<MonitoringConfiguration, ConfigurationError> {
        let mut config = Self::default_configuration();
        config.metadata.source = ConfigurationSource::Environment;

        // Parse environment variables and map to configuration fields
        self.apply_environment_overrides(&mut config)?;

        // Validate configuration
        self.validate_configuration(&config)?;

        Ok(config)
    }

    /// Merge multiple configurations
    pub fn merge_configurations(&self, configs: Vec<MonitoringConfiguration>) -> Result<MonitoringConfiguration, ConfigurationError> {
        if configs.is_empty() {
            return Ok(Self::default_configuration());
        }

        let mut merged = configs[0].clone();
        let sources: Vec<ConfigurationSource> = configs.iter()
            .map(|c| c.metadata.source.clone())
            .collect();

        for config in configs.iter().skip(1) {
            merged = self.merge_two_configurations(merged, config.clone())?;
        }

        merged.metadata.source = ConfigurationSource::Merged(sources);
        merged.metadata.modified_at = Utc::now();

        // Validate merged configuration
        self.validate_configuration(&merged)?;

        Ok(merged)
    }

    /// Validate configuration against schema
    pub fn validate_configuration(&self, config: &MonitoringConfiguration) -> Result<(), ConfigurationError> {
        let mut errors = Vec::new();

        // Validate against schema
        for (field, validator) in &self.validation_schema.validators {
            if let Some(value) = self.get_field_value(config, field) {
                if let Err(error) = validator.validate(value) {
                    errors.push(error);
                }
            }
        }

        // Check required fields
        for required_field in &self.validation_schema.required_fields {
            if self.get_field_value(config, required_field).is_none() {
                errors.push(ValidationError {
                    field: required_field.clone(),
                    message: "Required field is missing".to_string(),
                    code: "REQUIRED_FIELD_MISSING".to_string(),
                    suggestion: Some(format!("Add the required field: {}", required_field)),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigurationError::ValidationFailed(errors))
        }
    }

    /// Update current configuration
    pub fn update_configuration(&self, new_config: MonitoringConfiguration) -> Result<(), ConfigurationError> {
        // Validate new configuration
        self.validate_configuration(&new_config)?;

        // Get old configuration for change notification
        let old_config = self.current_config.read().unwrap().clone();

        // Update configuration
        {
            let mut config = self.current_config.write().unwrap();
            *config = new_config.clone();
        }

        // Cache updated configuration
        self.cache_configuration(&new_config);

        // Record version if versioning is enabled
        if self.versioning.enabled {
            self.versioning.record_version(&old_config, &new_config);
        }

        // Notify listeners
        self.notify_configuration_changed(&old_config, &new_config);

        Ok(())
    }

    /// Get current configuration
    pub fn get_configuration(&self) -> MonitoringConfiguration {
        self.current_config.read().unwrap().clone()
    }

    /// Add configuration change listener
    pub fn add_listener(&self, listener: Box<dyn ConfigurationChangeListener + Send + Sync>) {
        self.listeners.lock().unwrap().push(listener);
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigurationError> {
        let config = self.get_configuration();

        // Encrypt if needed
        let config_to_save = if self.encryption.enabled {
            self.encryption.encrypt_configuration(config)?
        } else {
            config
        };

        // Serialize configuration
        let contents = serde_json::to_string_pretty(&config_to_save)
            .map_err(|e| ConfigurationError::InvalidFormat(e.to_string()))?;

        // Write to file
        fs::write(path, contents)
            .map_err(|e| ConfigurationError::Other(e.to_string()))?;

        Ok(())
    }

    /// Apply configuration template
    pub fn apply_template(&self, template_name: &str, parameters: Map<String, Value>) -> Result<MonitoringConfiguration, ConfigurationError> {
        // Load template configuration
        let template_config = self.load_template(template_name)?;

        // Apply template parameters
        let mut config = template_config;
        self.apply_template_parameters(&mut config, parameters)?;

        // Validate templated configuration
        self.validate_configuration(&config)?;

        Ok(config)
    }

    /// Get configuration diff between two configurations
    pub fn get_configuration_diff(&self, config1: &MonitoringConfiguration, config2: &MonitoringConfiguration) -> Vec<ConfigurationChange> {
        let mut changes = Vec::new();

        // Compare configurations and identify changes
        self.compare_configurations("", config1, config2, &mut changes);

        changes
    }

    /// Export configuration with specified format
    pub fn export_configuration(&self, format: &str) -> Result<String, ConfigurationError> {
        let config = self.get_configuration();

        match format.to_lowercase().as_str() {
            "json" => {
                serde_json::to_string_pretty(&config)
                    .map_err(|e| ConfigurationError::InvalidFormat(e.to_string()))
            },
            "yaml" => {
                // Note: Would need serde_yaml dependency
                Err(ConfigurationError::Other("YAML export not implemented".to_string()))
            },
            "toml" => {
                // Note: Would need toml dependency
                Err(ConfigurationError::Other("TOML export not implemented".to_string()))
            },
            _ => Err(ConfigurationError::InvalidFormat(format!("Unsupported format: {}", format))),
        }
    }

    /// Import configuration from string with specified format
    pub fn import_configuration(&self, content: &str, format: &str) -> Result<MonitoringConfiguration, ConfigurationError> {
        let config = match format.to_lowercase().as_str() {
            "json" => {
                serde_json::from_str(content)
                    .map_err(|e| ConfigurationError::InvalidFormat(e.to_string()))?
            },
            "yaml" => {
                // Note: Would need serde_yaml dependency
                return Err(ConfigurationError::Other("YAML import not implemented".to_string()));
            },
            "toml" => {
                // Note: Would need toml dependency
                return Err(ConfigurationError::Other("TOML import not implemented".to_string()));
            },
            _ => return Err(ConfigurationError::InvalidFormat(format!("Unsupported format: {}", format))),
        };

        // Validate imported configuration
        self.validate_configuration(&config)?;

        Ok(config)
    }

    // Private helper methods

    fn default_configuration() -> MonitoringConfiguration {
        MonitoringConfiguration {
            metadata: ConfigurationMetadata {
                version: "1.0.0".to_string(),
                created_at: Utc::now(),
                modified_at: Utc::now(),
                checksum: "".to_string(),
                source: ConfigurationSource::Default,
                tags: vec!["default".to_string()],
                description: Some("Default monitoring configuration".to_string()),
                author: Some("system".to_string()),
            },
            monitoring: MonitoringSettings {
                enabled: true,
                sampling_rate: 1.0,
                max_sessions: 100,
                session_timeout: Duration::from_secs(3600),
                thread_pool_size: 4,
                buffer_sizes: BufferSizes {
                    events: 10000,
                    metrics: 5000,
                    alerts: 1000,
                    health_checks: 1000,
                },
                logging: LoggingConfiguration {
                    level: "info".to_string(),
                    format: "json".to_string(),
                    rotation: LogRotationSettings {
                        max_size: 104857600, // 100MB
                        max_files: 10,
                        compress: true,
                    },
                    filters: vec![],
                },
            },
            metrics: MetricsConfiguration {
                enabled: true,
                collection_interval: Duration::from_secs(60),
                aggregation_window: Duration::from_secs(300),
                max_memory_metrics: 100000,
                persistence: MetricsPersistenceConfig {
                    enabled: false,
                    backend: "file".to_string(),
                    backend_config: Map::new(),
                    persistence_interval: Duration::from_secs(300),
                },
                custom_metrics: vec![],
            },
            events: EventConfiguration {
                enabled: true,
                buffer_size: 10000,
                batch_size: 100,
                processing_interval: Duration::from_secs(10),
                filters: vec![],
                enrichment: EventEnrichmentConfig {
                    enabled: false,
                    sources: vec![],
                    rules: vec![],
                },
            },
            alerts: AlertConfiguration {
                enabled: true,
                evaluation_interval: Duration::from_secs(60),
                rules: vec![],
                channels: vec![],
                suppression: AlertSuppressionConfig {
                    enabled: false,
                    rules: vec![],
                    default_duration: Duration::from_secs(3600),
                },
            },
            health: HealthConfiguration {
                enabled: true,
                check_interval: Duration::from_secs(30),
                check_timeout: Duration::from_secs(5),
                checks: vec![],
                thresholds: HealthThresholds {
                    cpu_usage: 80.0,
                    memory_usage: 90.0,
                    disk_usage: 85.0,
                    network_latency: Duration::from_millis(100),
                    error_rate: 5.0,
                },
            },
            performance: PerformanceConfiguration {
                enabled: true,
                analysis_interval: Duration::from_secs(300),
                window_size: Duration::from_secs(3600),
                anomaly_detection: AnomalyDetectionConfig {
                    enabled: false,
                    algorithms: vec!["statistical".to_string()],
                    sensitivity: 0.5,
                    min_data_points: 50,
                },
                trend_analysis: TrendAnalysisConfig {
                    enabled: false,
                    methods: vec!["linear".to_string()],
                    significance_threshold: 0.05,
                },
            },
            reporting: ReportingConfiguration {
                enabled: false,
                generation_interval: Duration::from_secs(86400), // 24 hours
                templates: vec![],
                distribution: ReportDistributionConfig {
                    enabled: false,
                    channels: vec![],
                    schedule: "daily".to_string(),
                    filters: vec![],
                },
            },
            retention: RetentionConfiguration {
                enabled: true,
                default_retention: Duration::from_secs(2592000), // 30 days
                policies: vec![],
                cleanup_schedule: "0 2 * * *".to_string(), // Daily at 2 AM
            },
            security: SecurityConfiguration {
                enabled: false,
                encryption: EncryptionConfig {
                    enabled: false,
                    algorithm: "AES-256-GCM".to_string(),
                    key_management: KeyManagementConfig {
                        provider: "local".to_string(),
                        rotation_enabled: false,
                        rotation_interval: Duration::from_secs(2592000), // 30 days
                        config: Map::new(),
                    },
                    scope: vec![],
                },
                authentication: AuthenticationConfig {
                    enabled: false,
                    methods: vec!["bearer".to_string()],
                    token: TokenConfig {
                        token_type: "JWT".to_string(),
                        expiration: Duration::from_secs(3600),
                        refresh_enabled: true,
                    },
                },
                authorization: AuthorizationConfig {
                    enabled: false,
                    model: "rbac".to_string(),
                    permissions: vec![],
                },
                audit: AuditConfig {
                    enabled: false,
                    events: vec![],
                    format: "json".to_string(),
                    retention: Duration::from_secs(7776000), // 90 days
                },
            },
            features: FeatureConfiguration {
                flags: HashMap::new(),
                refresh_interval: Duration::from_secs(300),
                remote: None,
            },
            environment: EnvironmentConfiguration {
                environment: "development".to_string(),
                overrides: Map::new(),
                variables: HashMap::new(),
                validation: vec![],
            },
            plugins: PluginConfiguration {
                discovery_enabled: false,
                plugin_directories: vec![],
                plugins: vec![],
                security: PluginSecurityConfig {
                    signature_verification: true,
                    allowed_sources: vec![],
                    sandboxing: true,
                    resource_limits: PluginResourceLimits {
                        max_memory: 104857600, // 100MB
                        max_cpu: 50.0,
                        max_file_handles: 100,
                        max_network_connections: 10,
                    },
                },
            },
        }
    }

    fn default_schema() -> ConfigurationSchema {
        ConfigurationSchema {
            version: "1.0.0".to_string(),
            validators: HashMap::new(),
            required_fields: HashSet::from([
                "metadata.version".to_string(),
                "monitoring.enabled".to_string(),
            ]),
            defaults: Map::new(),
        }
    }

    fn apply_environment_overrides(&self, config: &mut MonitoringConfiguration) -> Result<(), ConfigurationError> {
        // Apply environment variable overrides to configuration
        // This would parse environment variables and map them to configuration fields
        Ok(())
    }

    fn merge_two_configurations(&self, base: MonitoringConfiguration, overlay: MonitoringConfiguration) -> Result<MonitoringConfiguration, ConfigurationError> {
        // Merge two configurations with overlay taking precedence
        // This is a simplified implementation - a real implementation would do deep merging
        Ok(overlay)
    }

    fn get_field_value(&self, config: &MonitoringConfiguration, field_path: &str) -> Option<&Value> {
        // Navigate to the specified field using dot notation
        // This is a simplified implementation
        None
    }

    fn cache_configuration(&self, config: &MonitoringConfiguration) {
        let mut cache = self.cache.lock().unwrap();
        let cache_key = format!("{:?}", config.metadata.source);

        cache.configs.insert(cache_key, CachedConfiguration {
            config: config.clone(),
            cached_at: SystemTime::now(),
            ttl: Duration::from_secs(3600),
            hit_count: 0,
        });
    }

    fn notify_configuration_changed(&self, old_config: &MonitoringConfiguration, new_config: &MonitoringConfiguration) {
        let listeners = self.listeners.lock().unwrap();
        for listener in listeners.iter() {
            listener.on_configuration_changed(old_config, new_config);
        }
    }

    fn load_template(&self, template_name: &str) -> Result<MonitoringConfiguration, ConfigurationError> {
        // Load configuration template
        // This would load from a template registry or file system
        Err(ConfigurationError::Other("Template loading not implemented".to_string()))
    }

    fn apply_template_parameters(&self, config: &mut MonitoringConfiguration, parameters: Map<String, Value>) -> Result<(), ConfigurationError> {
        // Apply template parameters to configuration
        // This would substitute template variables with actual values
        Ok(())
    }

    fn compare_configurations(&self, prefix: &str, config1: &MonitoringConfiguration, config2: &MonitoringConfiguration, changes: &mut Vec<ConfigurationChange>) {
        // Compare two configurations and identify changes
        // This is a simplified implementation
    }
}

impl ConfigurationCache {
    fn new() -> Self {
        Self {
            configs: HashMap::new(),
            stats: CacheStatistics::default(),
            settings: CacheSettings {
                max_size: 100,
                default_ttl: Duration::from_secs(3600),
                enabled: true,
            },
        }
    }
}

impl ConfigurationEncryption {
    fn new() -> Self {
        Self {
            algorithm: "AES-256-GCM".to_string(),
            key: vec![0u8; 32], // In real implementation, this would be properly generated
            enabled: false,
        }
    }

    fn encrypt_configuration(&self, config: MonitoringConfiguration) -> Result<MonitoringConfiguration, ConfigurationError> {
        // Encrypt sensitive configuration fields
        // This is a placeholder - real implementation would use proper encryption
        Ok(config)
    }

    fn decrypt_configuration(&self, config: MonitoringConfiguration) -> Result<MonitoringConfiguration, ConfigurationError> {
        // Decrypt sensitive configuration fields
        // This is a placeholder - real implementation would use proper decryption
        Ok(config)
    }
}

impl ConfigurationVersioning {
    fn new() -> Self {
        Self {
            history: Vec::new(),
            current_version: "1.0.0".to_string(),
            enabled: false,
        }
    }

    fn record_version(&self, old_config: &MonitoringConfiguration, new_config: &MonitoringConfiguration) {
        // Record configuration version
        // This would create a version entry and store the changes
    }
}

impl Default for ConfigurationManager {
    fn default() -> Self {
        Self::new()
    }
}

// Custom serialization for Duration
fn serialize_duration<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_u64(duration.as_secs())
}

fn deserialize_duration<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: Deserializer<'de>,
{
    let secs = u64::deserialize(deserializer)?;
    Ok(Duration::from_secs(secs))
}

// Re-export common types for convenience
pub use self::{
    ConfigurationManager,
    MonitoringConfiguration,
    ConfigurationError,
    ConfigurationChangeListener,
    ConfigurationValidator,
    ValidationError,
    ValidationConstraint,
};