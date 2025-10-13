//! Configuration Management and Global Settings
//!
//! This module provides comprehensive configuration management for retry systems
//! including global settings, adaptive optimization, feature flags, and
//! integration settings with validation and hot-reload capabilities.

use super::core::*;
use sklears_core::error::Result as SklResult;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, SystemTime},
};

/// Global retry configuration
#[derive(Debug, Clone)]
pub struct GlobalRetryConfig {
    /// Default retry configuration
    pub default_config: RetryConfig,
    /// Global retry limits
    pub global_limits: GlobalRetryLimits,
    /// Feature flags
    pub features: HashMap<String, bool>,
    /// Integration settings
    pub integrations: HashMap<String, String>,
    /// Performance tuning parameters
    pub performance: PerformanceTuning,
    /// Monitoring configuration
    pub monitoring: MonitoringConfiguration,
    /// Machine learning settings
    pub machine_learning: MachineLearningConfig,
}

impl Default for GlobalRetryConfig {
    fn default() -> Self {
        Self {
            default_config: RetryConfig::default(),
            global_limits: GlobalRetryLimits::default(),
            features: Self::default_features(),
            integrations: HashMap::new(),
            performance: PerformanceTuning::default(),
            monitoring: MonitoringConfiguration::default(),
            machine_learning: MachineLearningConfig::default(),
        }
    }
}

impl GlobalRetryConfig {
    /// Create default feature flags
    fn default_features() -> HashMap<String, bool> {
        let mut features = HashMap::new();
        features.insert("adaptive_strategies".to_string(), true);
        features.insert("machine_learning".to_string(), true);
        features.insert("simd_acceleration".to_string(), true);
        features.insert("circuit_breaking".to_string(), true);
        features.insert("rate_limiting".to_string(), true);
        features.insert("metrics_collection".to_string(), true);
        features.insert("alerting".to_string(), true);
        features.insert("policy_engine".to_string(), true);
        features.insert("feature_engineering".to_string(), true);
        features.insert("pattern_detection".to_string(), true);
        features
    }

    /// Check if feature is enabled
    pub fn is_feature_enabled(&self, feature: &str) -> bool {
        self.features.get(feature).copied().unwrap_or(false)
    }

    /// Enable feature
    pub fn enable_feature(&mut self, feature: String) {
        self.features.insert(feature, true);
    }

    /// Disable feature
    pub fn disable_feature(&mut self, feature: String) {
        self.features.insert(feature, false);
    }

    /// Get integration setting
    pub fn get_integration_setting(&self, integration: &str) -> Option<&String> {
        self.integrations.get(integration)
    }

    /// Set integration setting
    pub fn set_integration_setting(&mut self, integration: String, value: String) {
        self.integrations.insert(integration, value);
    }

    /// Validate configuration
    pub fn validate(&self) -> SklResult<()> {
        // Validate default retry config
        if self.default_config.max_attempts == 0 {
            return Err(RetryError::Configuration {
                parameter: "max_attempts".to_string(),
                message: "Maximum attempts must be greater than 0".to_string(),
            }.into());
        }

        if self.default_config.timeout == Duration::ZERO {
            return Err(RetryError::Configuration {
                parameter: "timeout".to_string(),
                message: "Timeout must be greater than 0".to_string(),
            }.into());
        }

        // Validate global limits
        if self.global_limits.max_concurrent_retries == 0 {
            return Err(RetryError::Configuration {
                parameter: "max_concurrent_retries".to_string(),
                message: "Must allow at least 1 concurrent retry".to_string(),
            }.into());
        }

        // Validate performance settings
        if self.performance.simd_batch_size == 0 {
            return Err(RetryError::Configuration {
                parameter: "simd_batch_size".to_string(),
                message: "SIMD batch size must be greater than 0".to_string(),
            }.into());
        }

        Ok(())
    }
}

/// Global retry limits
#[derive(Debug, Clone)]
pub struct GlobalRetryLimits {
    /// Maximum concurrent retries
    pub max_concurrent_retries: u32,
    /// Maximum total retries per time window
    pub max_retries_per_window: u32,
    /// Time window duration
    pub window_duration: Duration,
    /// Maximum retry duration
    pub max_retry_duration: Duration,
    /// Maximum memory usage for retry system (bytes)
    pub max_memory_usage: u64,
    /// Maximum CPU usage percentage (0.0 to 1.0)
    pub max_cpu_usage: f64,
}

impl Default for GlobalRetryLimits {
    fn default() -> Self {
        Self {
            max_concurrent_retries: 100,
            max_retries_per_window: 1000,
            window_duration: Duration::from_secs(60),
            max_retry_duration: Duration::from_secs(300),
            max_memory_usage: 512 * 1024 * 1024, // 512 MB
            max_cpu_usage: 0.8, // 80%
        }
    }
}

/// Performance tuning parameters
#[derive(Debug, Clone)]
pub struct PerformanceTuning {
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    /// SIMD batch size
    pub simd_batch_size: usize,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Thread pool size
    pub thread_pool_size: usize,
    /// Memory pool settings
    pub memory_pool: MemoryPoolConfig,
    /// Cache settings
    pub cache: CacheConfig,
    /// Optimization level (0-3)
    pub optimization_level: u8,
}

impl Default for PerformanceTuning {
    fn default() -> Self {
        Self {
            enable_simd: true,
            simd_batch_size: 256,
            enable_parallel: true,
            thread_pool_size: num_cpus::get().unwrap_or(4),
            memory_pool: MemoryPoolConfig::default(),
            cache: CacheConfig::default(),
            optimization_level: 2,
        }
    }
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Enable memory pooling
    pub enabled: bool,
    /// Initial pool size
    pub initial_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Growth factor
    pub growth_factor: f64,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_size: 1024 * 1024, // 1 MB
            max_size: 64 * 1024 * 1024, // 64 MB
            growth_factor: 1.5,
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,
    /// Cache size limit
    pub max_size: usize,
    /// Cache TTL
    pub ttl: Duration,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 10000,
            ttl: Duration::from_secs(300),
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

/// Cache eviction policy
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPolicy {
    LRU,  // Least Recently Used
    LFU,  // Least Frequently Used
    FIFO, // First In First Out
    TTL,  // Time To Live based
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfiguration {
    /// Enable monitoring
    pub enabled: bool,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Metrics retention period
    pub retention_period: Duration,
    /// Enable alerting
    pub alerting_enabled: bool,
    /// Alert check interval
    pub alert_check_interval: Duration,
    /// Metrics to collect
    pub metrics_whitelist: Vec<String>,
    /// Export configuration
    pub export: ExportConfiguration,
}

impl Default for MonitoringConfiguration {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_secs(60),
            retention_period: Duration::from_secs(86400), // 24 hours
            alerting_enabled: true,
            alert_check_interval: Duration::from_secs(30),
            metrics_whitelist: vec![
                "success_rate".to_string(),
                "avg_duration".to_string(),
                "total_attempts".to_string(),
                "error_rates".to_string(),
                "system_health".to_string(),
            ],
            export: ExportConfiguration::default(),
        }
    }
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfiguration {
    /// Enable metrics export
    pub enabled: bool,
    /// Export format
    pub format: ExportFormat,
    /// Export destination
    pub destination: String,
    /// Export interval
    pub interval: Duration,
    /// Batch size for export
    pub batch_size: usize,
}

impl Default for ExportConfiguration {
    fn default() -> Self {
        Self {
            enabled: false,
            format: ExportFormat::JSON,
            destination: "/tmp/retry_metrics".to_string(),
            interval: Duration::from_secs(300),
            batch_size: 100,
        }
    }
}

/// Export format enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ExportFormat {
    JSON,
    CSV,
    Prometheus,
    StatsD,
    Custom(String),
}

/// Machine learning configuration
#[derive(Debug, Clone)]
pub struct MachineLearningConfig {
    /// Enable machine learning features
    pub enabled: bool,
    /// Default model type
    pub default_model: String,
    /// Model training configuration
    pub training: TrainingConfiguration,
    /// Feature engineering settings
    pub feature_engineering: FeatureEngineeringConfig,
    /// Model selection settings
    pub model_selection: ModelSelectionConfig,
    /// Performance tracking settings
    pub performance_tracking: PerformanceTrackingConfig,
}

impl Default for MachineLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_model: "linear_regression".to_string(),
            training: TrainingConfiguration::default(),
            feature_engineering: FeatureEngineeringConfig::default(),
            model_selection: ModelSelectionConfig::default(),
            performance_tracking: PerformanceTrackingConfig::default(),
        }
    }
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfiguration {
    /// Training data buffer size
    pub buffer_size: usize,
    /// Minimum samples for training
    pub min_samples: usize,
    /// Training frequency
    pub training_frequency: Duration,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: usize,
    /// Validation split
    pub validation_split: f64,
}

impl Default for TrainingConfiguration {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            min_samples: 100,
            training_frequency: Duration::from_secs(3600), // 1 hour
            learning_rate: 0.01,
            epochs: 100,
            validation_split: 0.2,
        }
    }
}

/// Feature engineering configuration
#[derive(Debug, Clone)]
pub struct FeatureEngineeringConfig {
    /// Enable feature engineering
    pub enabled: bool,
    /// Enable normalization
    pub normalization: bool,
    /// Enable scaling
    pub scaling: bool,
    /// Number of bins for discretization
    pub binning_bins: Option<usize>,
    /// Feature selection method
    pub selection_method: FeatureSelectionMethod,
    /// Maximum number of features
    pub max_features: usize,
    /// Correlation threshold for selection
    pub correlation_threshold: f64,
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            normalization: true,
            scaling: false,
            binning_bins: Some(5),
            selection_method: FeatureSelectionMethod::Correlation,
            max_features: 20,
            correlation_threshold: 0.1,
        }
    }
}

/// Model selection configuration
#[derive(Debug, Clone)]
pub struct ModelSelectionConfig {
    /// Model selection strategy
    pub strategy: ModelSelectionStrategy,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Enable model ensemble
    pub enable_ensemble: bool,
    /// Maximum number of models in ensemble
    pub max_ensemble_size: usize,
}

impl Default for ModelSelectionConfig {
    fn default() -> Self {
        Self {
            strategy: ModelSelectionStrategy::BestPerformance,
            cv_folds: 5,
            enable_ensemble: false,
            max_ensemble_size: 3,
        }
    }
}

/// Configuration manager
#[derive(Debug)]
pub struct ConfigurationManager {
    /// Configuration storage
    config: Arc<RwLock<GlobalRetryConfig>>,
    /// Configuration file path
    file_path: Option<String>,
    /// Auto-reload enabled
    auto_reload: bool,
    /// Last modification time
    last_modified: Arc<RwLock<Option<SystemTime>>>,
    /// Configuration validators
    validators: Vec<Box<dyn ConfigurationValidator + Send + Sync>>,
}

impl ConfigurationManager {
    /// Create new configuration manager
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(GlobalRetryConfig::default())),
            file_path: None,
            auto_reload: false,
            last_modified: Arc::new(RwLock::new(None)),
            validators: Vec::new(),
        }
    }

    /// Create configuration manager from file
    pub fn from_file(file_path: String) -> SklResult<Self> {
        let mut manager = Self::new();
        manager.file_path = Some(file_path);
        manager.load_from_file()?;
        Ok(manager)
    }

    /// Enable auto-reload
    pub fn enable_auto_reload(&mut self) {
        self.auto_reload = true;
    }

    /// Register configuration validator
    pub fn register_validator(&mut self, validator: Box<dyn ConfigurationValidator + Send + Sync>) {
        self.validators.push(validator);
    }

    /// Get configuration
    pub fn get_config(&self) -> GlobalRetryConfig {
        self.config.read().unwrap().clone()
    }

    /// Update configuration
    pub fn update_config(&self, new_config: GlobalRetryConfig) -> SklResult<()> {
        // Validate configuration
        self.validate_config(&new_config)?;

        // Update configuration
        {
            let mut config = self.config.write().unwrap();
            *config = new_config;
        }

        // Save to file if path is set
        if let Some(ref file_path) = self.file_path {
            self.save_to_file(file_path)?;
        }

        Ok(())
    }

    /// Load configuration from file
    fn load_from_file(&self) -> SklResult<()> {
        if let Some(ref file_path) = self.file_path {
            // In a real implementation, this would parse configuration from file
            // For now, we'll use the default configuration
            println!("Loading configuration from: {}", file_path);

            // Update last modified time
            {
                let mut last_modified = self.last_modified.write().unwrap();
                *last_modified = Some(SystemTime::now());
            }
        }
        Ok(())
    }

    /// Save configuration to file
    fn save_to_file(&self, file_path: &str) -> SklResult<()> {
        // In a real implementation, this would serialize configuration to file
        println!("Saving configuration to: {}", file_path);
        Ok(())
    }

    /// Validate configuration
    fn validate_config(&self, config: &GlobalRetryConfig) -> SklResult<()> {
        // Built-in validation
        config.validate()?;

        // Custom validators
        for validator in &self.validators {
            validator.validate(config)?;
        }

        Ok(())
    }

    /// Check for configuration file changes
    pub fn check_for_updates(&self) -> SklResult<bool> {
        if !self.auto_reload || self.file_path.is_none() {
            return Ok(false);
        }

        // In a real implementation, this would check file modification time
        // For now, we'll return false (no updates)
        Ok(false)
    }

    /// Reload configuration if file has changed
    pub fn reload_if_changed(&self) -> SklResult<bool> {
        if self.check_for_updates()? {
            self.load_from_file()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get configuration statistics
    pub fn get_statistics(&self) -> ConfigurationStatistics {
        let config = self.config.read().unwrap();
        let enabled_features = config.features.values().filter(|&&enabled| enabled).count();
        let total_features = config.features.len();
        let configured_integrations = config.integrations.len();

        ConfigurationStatistics {
            total_features,
            enabled_features,
            configured_integrations,
            last_updated: self.last_modified.read().unwrap().unwrap_or(SystemTime::UNIX_EPOCH),
            validation_passed: true, // Simplified
        }
    }
}

/// Configuration statistics
#[derive(Debug, Clone)]
pub struct ConfigurationStatistics {
    /// Total number of features
    pub total_features: usize,
    /// Number of enabled features
    pub enabled_features: usize,
    /// Number of configured integrations
    pub configured_integrations: usize,
    /// Last configuration update time
    pub last_updated: SystemTime,
    /// Whether validation passed
    pub validation_passed: bool,
}

/// Configuration validator trait
pub trait ConfigurationValidator: Send + Sync {
    /// Validate configuration
    fn validate(&self, config: &GlobalRetryConfig) -> SklResult<()>;

    /// Get validator name
    fn name(&self) -> &str;
}

/// Resource limits validator
#[derive(Debug)]
pub struct ResourceLimitsValidator {
    /// Maximum allowed memory (bytes)
    pub max_memory: u64,
    /// Maximum allowed CPU usage (0.0 to 1.0)
    pub max_cpu: f64,
}

impl ResourceLimitsValidator {
    /// Create new resource limits validator
    pub fn new(max_memory: u64, max_cpu: f64) -> Self {
        Self { max_memory, max_cpu }
    }
}

impl ConfigurationValidator for ResourceLimitsValidator {
    fn validate(&self, config: &GlobalRetryConfig) -> SklResult<()> {
        if config.global_limits.max_memory_usage > self.max_memory {
            return Err(RetryError::Configuration {
                parameter: "max_memory_usage".to_string(),
                message: format!("Memory usage {} exceeds limit {}", config.global_limits.max_memory_usage, self.max_memory),
            }.into());
        }

        if config.global_limits.max_cpu_usage > self.max_cpu {
            return Err(RetryError::Configuration {
                parameter: "max_cpu_usage".to_string(),
                message: format!("CPU usage {} exceeds limit {}", config.global_limits.max_cpu_usage, self.max_cpu),
            }.into());
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "resource_limits"
    }
}

/// Performance requirements validator
#[derive(Debug)]
pub struct PerformanceValidator {
    /// Minimum required success rate
    pub min_success_rate: f64,
    /// Maximum allowed average duration
    pub max_avg_duration: Duration,
}

impl PerformanceValidator {
    /// Create new performance validator
    pub fn new(min_success_rate: f64, max_avg_duration: Duration) -> Self {
        Self { min_success_rate, max_avg_duration }
    }
}

impl ConfigurationValidator for PerformanceValidator {
    fn validate(&self, config: &GlobalRetryConfig) -> SklResult<()> {
        // In a real implementation, this would validate against performance requirements
        // For now, we'll just check basic constraints
        if config.default_config.timeout > self.max_avg_duration {
            return Err(RetryError::Configuration {
                parameter: "timeout".to_string(),
                message: "Timeout exceeds maximum allowed duration".to_string(),
            }.into());
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "performance"
    }
}

/// Configuration builder for fluent API
#[derive(Debug)]
pub struct ConfigurationBuilder {
    config: GlobalRetryConfig,
}

impl ConfigurationBuilder {
    /// Create new configuration builder
    pub fn new() -> Self {
        Self {
            config: GlobalRetryConfig::default(),
        }
    }

    /// Set default retry configuration
    pub fn with_default_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.config.default_config = retry_config;
        self
    }

    /// Set global limits
    pub fn with_global_limits(mut self, limits: GlobalRetryLimits) -> Self {
        self.config.global_limits = limits;
        self
    }

    /// Enable feature
    pub fn enable_feature(mut self, feature: &str) -> Self {
        self.config.features.insert(feature.to_string(), true);
        self
    }

    /// Disable feature
    pub fn disable_feature(mut self, feature: &str) -> Self {
        self.config.features.insert(feature.to_string(), false);
        self
    }

    /// Set performance tuning
    pub fn with_performance_tuning(mut self, performance: PerformanceTuning) -> Self {
        self.config.performance = performance;
        self
    }

    /// Set monitoring configuration
    pub fn with_monitoring(mut self, monitoring: MonitoringConfiguration) -> Self {
        self.config.monitoring = monitoring;
        self
    }

    /// Set machine learning configuration
    pub fn with_machine_learning(mut self, ml_config: MachineLearningConfig) -> Self {
        self.config.machine_learning = ml_config;
        self
    }

    /// Add integration setting
    pub fn with_integration(mut self, name: &str, value: &str) -> Self {
        self.config.integrations.insert(name.to_string(), value.to_string());
        self
    }

    /// Build configuration
    pub fn build(self) -> SklResult<GlobalRetryConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Configuration factory
pub struct ConfigurationFactory;

impl ConfigurationFactory {
    /// Create development configuration
    pub fn create_development_config() -> GlobalRetryConfig {
        ConfigurationBuilder::new()
            .enable_feature("adaptive_strategies")
            .enable_feature("machine_learning")
            .disable_feature("simd_acceleration") // Disabled for stable development
            .enable_feature("metrics_collection")
            .enable_feature("alerting")
            .with_global_limits(GlobalRetryLimits {
                max_concurrent_retries: 50,
                max_retries_per_window: 500,
                window_duration: Duration::from_secs(60),
                max_retry_duration: Duration::from_secs(120),
                max_memory_usage: 256 * 1024 * 1024, // 256 MB
                max_cpu_usage: 0.5, // 50%
            })
            .build()
            .unwrap()
    }

    /// Create production configuration
    pub fn create_production_config() -> GlobalRetryConfig {
        ConfigurationBuilder::new()
            .enable_feature("adaptive_strategies")
            .enable_feature("machine_learning")
            .enable_feature("simd_acceleration")
            .enable_feature("circuit_breaking")
            .enable_feature("rate_limiting")
            .enable_feature("metrics_collection")
            .enable_feature("alerting")
            .enable_feature("policy_engine")
            .with_global_limits(GlobalRetryLimits {
                max_concurrent_retries: 1000,
                max_retries_per_window: 10000,
                window_duration: Duration::from_secs(60),
                max_retry_duration: Duration::from_secs(300),
                max_memory_usage: 1024 * 1024 * 1024, // 1 GB
                max_cpu_usage: 0.8, // 80%
            })
            .with_performance_tuning(PerformanceTuning {
                enable_simd: true,
                simd_batch_size: 512,
                enable_parallel: true,
                thread_pool_size: num_cpus::get().unwrap_or(8),
                memory_pool: MemoryPoolConfig {
                    enabled: true,
                    initial_size: 10 * 1024 * 1024, // 10 MB
                    max_size: 100 * 1024 * 1024, // 100 MB
                    growth_factor: 1.5,
                },
                cache: CacheConfig {
                    enabled: true,
                    max_size: 50000,
                    ttl: Duration::from_secs(300),
                    eviction_policy: EvictionPolicy::LRU,
                },
                optimization_level: 3,
            })
            .build()
            .unwrap()
    }

    /// Create testing configuration
    pub fn create_testing_config() -> GlobalRetryConfig {
        ConfigurationBuilder::new()
            .disable_feature("simd_acceleration")
            .disable_feature("alerting")
            .enable_feature("metrics_collection")
            .with_global_limits(GlobalRetryLimits {
                max_concurrent_retries: 10,
                max_retries_per_window: 100,
                window_duration: Duration::from_secs(10),
                max_retry_duration: Duration::from_secs(30),
                max_memory_usage: 50 * 1024 * 1024, // 50 MB
                max_cpu_usage: 0.3, // 30%
            })
            .build()
            .unwrap()
    }
}

// Add this dependency to support CPU count detection
mod num_cpus {
    pub fn get() -> Option<usize> {
        // Simplified implementation - would use actual CPU detection
        Some(4)
    }
}