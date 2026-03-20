use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::alerting_engine::{AlertingImplementationConfig, RuleEngineType, OptimizationTechnique};
use super::dashboard_management::DashboardImplementationConfig;
use super::performance_monitoring::PerformanceMonitoringConfig;
use super::recovery_systems::{RecoveryConfig, ClusterType};
use super::runtime_configuration::RuntimeConfig;
use super::validation_framework::{ValidationConfig, ValidationErrorHandling};

/// Comprehensive alerting, performance, and validation implementation configuration
/// Consolidates all enterprise features including alerting, rule engines, performance monitoring,
/// validation, recovery, and system configuration across multiple specialized modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingPerformanceValidationConfig {
    /// Alerting implementation configuration
    pub alerting_implementation: AlertingImplementationConfig,
    /// Dashboard implementation configuration
    pub dashboard_implementation: DashboardImplementationConfig,
    /// Performance monitoring configuration
    pub performance_monitoring: PerformanceMonitoringConfig,
    /// Recovery configuration
    pub recovery: RecoveryConfig,
    /// Validation configuration
    pub validation: ValidationConfig,
    /// Runtime configuration
    pub runtime: RuntimeConfig,
}

impl Default for AlertingPerformanceValidationConfig {
    fn default() -> Self {
        Self {
            alerting_implementation: AlertingImplementationConfig::default(),
            dashboard_implementation: DashboardImplementationConfig::default(),
            performance_monitoring: PerformanceMonitoringConfig::default(),
            recovery: RecoveryConfig::default(),
            validation: ValidationConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }
}

impl AlertingPerformanceValidationConfig {
    /// Create a new configuration with enterprise defaults
    /// Optimized for high-performance production environments with comprehensive monitoring,
    /// advanced alerting, high availability clustering, and strict validation
    pub fn enterprise_default() -> Self {
        let mut config = Self::default();

        // Configure high-performance alerting
        config.alerting_implementation.rule_engine.engine_type = RuleEngineType::Hybrid;
        config.alerting_implementation.rule_engine.rule_execution.parallelization.enabled = true;
        config.alerting_implementation.rule_engine.rule_optimization.adaptive_optimization = true;
        config.alerting_implementation.rule_engine.rule_optimization.optimization_techniques.push(OptimizationTechnique::Vectorization);
        config.alerting_implementation.rule_engine.rule_optimization.optimization_techniques.push(OptimizationTechnique::Memoization);

        // Enable comprehensive monitoring
        config.performance_monitoring.cpu_monitoring.enabled = true;
        config.performance_monitoring.memory_monitoring.enabled = true;
        config.performance_monitoring.disk_monitoring.enabled = true;
        config.performance_monitoring.network_monitoring.enabled = true;
        config.performance_monitoring.application_monitoring.response_time.enabled = true;
        config.performance_monitoring.application_monitoring.error_rate.enabled = true;
        config.performance_monitoring.application_monitoring.throughput.enabled = true;

        // Configure high availability
        config.recovery.high_availability.clustering.cluster_type = ClusterType::ActiveActive;

        // Enable comprehensive validation
        config.validation.runtime_validation.performance_validation.enabled = true;
        config.validation.runtime_validation.resource_validation.enabled = true;
        config.validation.data_validation.consistency_validation.cross_system_consistency = true;
        config.validation.configuration_validation.security_validation.vulnerability_scanning = true;

        // Enhanced error handling
        config.runtime.error_handling.aggregation.window = Duration::from_secs(60); // 1 minute
        config.runtime.error_handling.aggregation.threshold = 50;

        config
    }

    /// Create a minimal configuration for development
    /// Optimized for development environments with simplified alerting, reduced monitoring overhead,
    /// lenient validation, and minimal resource requirements
    pub fn development_default() -> Self {
        let mut config = Self::default();

        // Simplified alerting for development
        config.alerting_implementation.rule_engine.engine_type = RuleEngineType::Simple;
        config.alerting_implementation.notification_engine.rate_limiting.enabled = false;
        config.alerting_implementation.rule_engine.rule_execution.parallelization.enabled = false;
        config.alerting_implementation.rule_engine.rule_optimization.adaptive_optimization = false;

        // Reduce monitoring overhead
        config.performance_monitoring.cpu_monitoring.interval = Duration::from_secs(300); // 5 minutes
        config.performance_monitoring.memory_monitoring.interval = Duration::from_secs(300);
        config.performance_monitoring.disk_monitoring.interval = Duration::from_secs(600); // 10 minutes
        config.performance_monitoring.network_monitoring.interval = Duration::from_secs(300);

        // Simplified validation
        config.validation.input_validation.error_handling = ValidationErrorHandling::Lenient;
        config.validation.runtime_validation.behavior_validation.enabled = false;
        config.validation.runtime_validation.performance_validation.enabled = false;
        config.validation.data_validation.consistency_validation.cross_system_consistency = false;
        config.validation.configuration_validation.security_validation.vulnerability_scanning = false;

        // Relaxed error handling
        config.runtime.error_handling.aggregation.window = Duration::from_secs(600); // 10 minutes
        config.runtime.error_handling.aggregation.threshold = 200;

        // Minimal resource allocation
        config.runtime.resource_allocation.cpu_allocation.cores = 2;
        config.runtime.resource_allocation.memory_allocation.limit_mb = 2048;

        config
    }

    /// Create a testing configuration
    /// Optimized for testing environments with enhanced logging, disabled external dependencies,
    /// and configuration suitable for automated testing pipelines
    pub fn testing_default() -> Self {
        let mut config = Self::development_default();

        // Disable external notifications for testing
        config.alerting_implementation.notification_engine.channels.clear();

        // Disable recovery testing to avoid side effects
        config.recovery.backup_recovery.testing.enabled = false;
        config.recovery.point_in_time_recovery.enabled = false;

        // Enable aggressive validation for testing
        config.validation.input_validation.error_handling = ValidationErrorHandling::Strict;
        config.validation.runtime_validation.performance_validation.enabled = true;
        config.validation.runtime_validation.performance_validation.regression_detection = true;

        // Faster monitoring intervals for testing
        config.performance_monitoring.cpu_monitoring.interval = Duration::from_secs(30);
        config.performance_monitoring.memory_monitoring.interval = Duration::from_secs(30);

        config
    }

    /// Create a high-security configuration
    /// Optimized for security-sensitive environments with enhanced validation, comprehensive auditing,
    /// strict access controls, and security-focused monitoring
    pub fn security_default() -> Self {
        let mut config = Self::enterprise_default();

        // Enhanced security validation
        config.validation.configuration_validation.security_validation.credential_validation = true;
        config.validation.configuration_validation.security_validation.permission_validation = true;
        config.validation.configuration_validation.security_validation.vulnerability_scanning = true;

        // Strict input validation
        config.validation.input_validation.error_handling = ValidationErrorHandling::Strict;
        config.validation.input_validation.sanitization = true;

        // Enhanced template security
        config.alerting_implementation.notification_engine.template_engine.security.sandbox_enabled = true;
        config.alerting_implementation.notification_engine.template_engine.security.forbidden_patterns.extend(vec![
            "script".to_string(),
            "javascript".to_string(),
            "eval".to_string(),
            "exec".to_string(),
        ]);

        // Secure backup and recovery
        config.recovery.point_in_time_recovery.log_shipping.encryption = true;
        config.recovery.point_in_time_recovery.log_shipping.compression = true;

        config
    }
}