//! Alert Management System - Modular Architecture
//!
//! This module provides a comprehensive alert management system for distributed optimization
//! environments, organized into focused submodules for maintainability and extensibility.
//!
//! # Architecture
//!
//! The alert management system is organized into five focused modules:
//!
//! - [`types_config`] - Shared types, enums, and configuration structures
//! - [`core_manager`] - Central AlertManager and system configuration
//! - [`rule_management`] - Alert rule management, templates, and dependencies
//! - [`state_tracking`] - Alert state management and lifecycle tracking
//! - [`evaluation_engine`] - Alert evaluation logic and processing
//!
//! # Quick Start
//!
//! ```rust
//! use crate::alert_management::{AlertManager, AlertRule, AlertSeverity};
//! use std::time::Duration;
//!
//! // Create a new alert manager
//! let manager = AlertManager::new();
//!
//! // Create and register an alert rule
//! let rule = AlertRule::builder()
//!     .rule_id("cpu_high")
//!     .name("High CPU Usage")
//!     .expression("cpu_usage > 80")
//!     .severity(AlertSeverity::High)
//!     .duration(Duration::from_secs(300))
//!     .build()?;
//!
//! manager.register_rule(rule)?;
//! ```
//!
//! # Features
//!
//! ## Core Alert Management
//! - Rule configuration and management
//! - Alert lifecycle tracking
//! - State management and transitions
//! - Evaluation engine with multiple strategies
//!
//! ## Advanced Features
//! - Dynamic thresholds with machine learning
//! - Multi-dimensional threshold evaluation
//! - Rule dependencies and templates
//! - Performance monitoring and optimization
//! - Alerting routing and escalation
//!
//! ## Performance Optimizations
//! - Parallel rule evaluation
//! - Evaluation caching
//! - Background processing
//! - Resource limit enforcement
//! - Load balancing across workers

pub mod types_config;
pub mod core_manager;
pub mod rule_management;
pub mod state_tracking;
pub mod evaluation_engine;

// Re-export core types and configurations
pub use types_config::{
    AlertSeverity, AlertThreshold, DynamicThreshold, DynamicMethod,
    MultiDimensionalThreshold, CombinationLogic, ThresholdAdjustmentRule,
    AdjustmentCondition, AdjustmentType, ThresholdOperator, TrendDirection,
    StatisticalComparison, EvaluationPriority, EvaluationResourceLimits,
    FallbackStrategy, EvaluationQoS, AlertRouting, RoutingRule, EscalationRule,
    EscalationCondition, PerformanceSnapshot, Weekday,
};

// Re-export core manager and configuration
pub use core_manager::{
    AlertManager, AlertManagerConfig, AlertGroupingConfig, AlertPerformanceConfig,
    AlertHistoryEntry, AlertHistoryState, AlertManagerStatistics,
    EvaluationPerformanceStats,
};

// Re-export rule management types
pub use rule_management::{
    AlertRule, AlertRuleMetadata, RuleSource, RuleLifecycleStage,
    AlertRuleDependency, DependencyType, DependencyCondition, RuleTemplateInfo,
    AlertRuleTemplate, TemplateParameter, TemplateParameterType,
    TemplateValidationRule, RuleValidationResult, RuleManager,
    TemplateEngine, RuleBuilderError,
};

// Re-export state tracking types
pub use state_tracking::{
    AlertStateTracker, StateTrackerConfig, AlertState, StateTransition,
    StateStatistics, AlertFrequencyTrends, FrequencyTrend,
    StateTrackerPerformance, ActiveAlert, StateTrackerSummary,
};

// Re-export evaluation engine types
pub use evaluation_engine::{
    AlertEvaluation, RecoveryCondition, EvaluationStrategy, StatisticalConfig,
    StatisticalMethod, MLConfig, MLModelType, TrainingRequirements,
    DataQualityRequirements, QualityThresholds, PredictionParameters,
    EnsembleParameters, AlertEvaluationEngine, EvaluationWorkerPool,
    WorkerLoadBalancer, WorkerHealthMonitor, EvaluationCache, CacheConfig,
    CacheStatistics, EvaluationScheduler, SchedulerConfig, ScheduleOptimizer,
    OptimizerMetrics, EvaluationPerformanceMonitor, PerformanceMonitorConfig,
};

/// Result type for alert management operations
pub type AlertResult<T> = Result<T, AlertError>;

/// Error types for alert management operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertError {
    /// Configuration error
    ConfigurationError(String),
    /// Rule validation error
    RuleValidationError(String),
    /// Evaluation error
    EvaluationError(String),
    /// State tracking error
    StateError(String),
    /// Template error
    TemplateError(String),
    /// Resource limit exceeded
    ResourceLimitExceeded(String),
    /// Network or connectivity error
    NetworkError(String),
    /// Serialization/deserialization error
    SerializationError(String),
    /// Generic internal error
    InternalError(String),
}

impl std::fmt::Display for AlertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            Self::RuleValidationError(msg) => write!(f, "Rule validation error: {}", msg),
            Self::EvaluationError(msg) => write!(f, "Evaluation error: {}", msg),
            Self::StateError(msg) => write!(f, "State tracking error: {}", msg),
            Self::TemplateError(msg) => write!(f, "Template error: {}", msg),
            Self::ResourceLimitExceeded(msg) => write!(f, "Resource limit exceeded: {}", msg),
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for AlertError {}

/// Convenience functions for creating common alert configurations
pub mod presets {
    use super::*;
    use std::time::Duration;

    /// Create a basic CPU usage alert rule
    pub fn cpu_usage_alert(threshold: f64, duration: Duration) -> AlertRule {
        AlertRule::builder()
            .rule_id("cpu_usage_high")
            .name("High CPU Usage")
            .description("Alert when CPU usage exceeds threshold")
            .expression(format!("cpu_usage > {}", threshold))
            .severity(AlertSeverity::High)
            .duration(duration)
            .threshold(AlertThreshold {
                operator: ThresholdOperator::GreaterThan,
                value: threshold,
                hysteresis: Some(5.0),
                percentage: None,
                dynamic_threshold: None,
                multi_dimensional: vec![],
                adjustment_rules: vec![],
            })
            .build()
            .expect("Failed to create CPU usage alert rule")
    }

    /// Create a basic memory usage alert rule
    pub fn memory_usage_alert(threshold: f64, duration: Duration) -> AlertRule {
        AlertRule::builder()
            .rule_id("memory_usage_high")
            .name("High Memory Usage")
            .description("Alert when memory usage exceeds threshold")
            .expression(format!("memory_usage > {}", threshold))
            .severity(AlertSeverity::High)
            .duration(duration)
            .threshold(AlertThreshold {
                operator: ThresholdOperator::GreaterThan,
                value: threshold,
                hysteresis: Some(5.0),
                percentage: None,
                dynamic_threshold: None,
                multi_dimensional: vec![],
                adjustment_rules: vec![],
            })
            .build()
            .expect("Failed to create memory usage alert rule")
    }

    /// Create a disk space alert rule
    pub fn disk_space_alert(threshold: f64, duration: Duration) -> AlertRule {
        AlertRule::builder()
            .rule_id("disk_space_low")
            .name("Low Disk Space")
            .description("Alert when available disk space falls below threshold")
            .expression(format!("disk_space_available < {}", threshold))
            .severity(AlertSeverity::Warning)
            .duration(duration)
            .threshold(AlertThreshold {
                operator: ThresholdOperator::LessThan,
                value: threshold,
                hysteresis: Some(5.0),
                percentage: Some(threshold),
                dynamic_threshold: None,
                multi_dimensional: vec![],
                adjustment_rules: vec![],
            })
            .build()
            .expect("Failed to create disk space alert rule")
    }

    /// Create a service availability alert rule
    pub fn service_availability_alert(service_name: &str, duration: Duration) -> AlertRule {
        AlertRule::builder()
            .rule_id(format!("{}_availability", service_name))
            .name(format!("{} Service Availability", service_name))
            .description(format!("Alert when {} service becomes unavailable", service_name))
            .expression(format!("service_up{{service=\"{}\"}} == 0", service_name))
            .severity(AlertSeverity::Critical)
            .duration(duration)
            .threshold(AlertThreshold {
                operator: ThresholdOperator::Equal,
                value: 0.0,
                hysteresis: None,
                percentage: None,
                dynamic_threshold: None,
                multi_dimensional: vec![],
                adjustment_rules: vec![],
            })
            .build()
            .expect("Failed to create service availability alert rule")
    }

    /// Create a response time alert rule
    pub fn response_time_alert(threshold_ms: f64, duration: Duration) -> AlertRule {
        AlertRule::builder()
            .rule_id("response_time_high")
            .name("High Response Time")
            .description("Alert when response time exceeds threshold")
            .expression(format!("response_time_ms > {}", threshold_ms))
            .severity(AlertSeverity::Medium)
            .duration(duration)
            .threshold(AlertThreshold {
                operator: ThresholdOperator::GreaterThan,
                value: threshold_ms,
                hysteresis: Some(threshold_ms * 0.1),
                percentage: None,
                dynamic_threshold: Some(DynamicThreshold {
                    method: DynamicMethod::MovingAverage,
                    window_size: Duration::from_secs(3600),
                    std_dev_multiplier: 2.0,
                    min_samples: 30,
                    update_frequency: Duration::from_secs(300),
                }),
                multi_dimensional: vec![],
                adjustment_rules: vec![],
            })
            .build()
            .expect("Failed to create response time alert rule")
    }
}

/// Utility functions for alert management operations
pub mod utils {
    use super::*;
    use std::collections::HashMap;

    /// Validate an alert rule configuration
    pub fn validate_alert_rule(rule: &AlertRule) -> AlertResult<()> {
        if rule.rule_id.is_empty() {
            return Err(AlertError::RuleValidationError("Rule ID cannot be empty".to_string()));
        }

        if rule.name.is_empty() {
            return Err(AlertError::RuleValidationError("Rule name cannot be empty".to_string()));
        }

        if rule.expression.is_empty() {
            return Err(AlertError::RuleValidationError("Expression cannot be empty".to_string()));
        }

        if rule.duration.as_secs() == 0 {
            return Err(AlertError::RuleValidationError("Duration must be greater than zero".to_string()));
        }

        Ok(())
    }

    /// Calculate alert priority score based on severity and other factors
    pub fn calculate_priority_score(
        severity: &AlertSeverity,
        duration: Duration,
        frequency: f64,
    ) -> f64 {
        let base_score = match severity {
            AlertSeverity::Critical => 100.0,
            AlertSeverity::High => 80.0,
            AlertSeverity::Medium => 60.0,
            AlertSeverity::Warning => 40.0,
            AlertSeverity::Info => 20.0,
            AlertSeverity::Low => 10.0,
            AlertSeverity::Debug => 5.0,
        };

        let duration_factor = 1.0 + (duration.as_secs() as f64 / 3600.0); // Hours
        let frequency_factor = 1.0 + (frequency / 10.0); // Per 10 minutes

        base_score * duration_factor * frequency_factor
    }

    /// Generate a unique alert fingerprint based on rule and context
    pub fn generate_alert_fingerprint(
        rule_id: &str,
        labels: &HashMap<String, String>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        rule_id.hash(&mut hasher);

        let mut sorted_labels: Vec<_> = labels.iter().collect();
        sorted_labels.sort_by_key(|(k, _)| *k);
        for (key, value) in sorted_labels {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }

    /// Check if two alerts should be grouped together
    pub fn should_group_alerts(
        alert1: &ActiveAlert,
        alert2: &ActiveAlert,
        grouping_config: &AlertGroupingConfig,
    ) -> bool {
        if !grouping_config.enable_label_grouping && !grouping_config.enable_time_grouping {
            return false;
        }

        if grouping_config.enable_time_grouping {
            let time_diff = alert1.created_at.duration_since(alert2.created_at)
                .unwrap_or(Duration::from_secs(0));
            if time_diff > grouping_config.time_window {
                return false;
            }
        }

        if grouping_config.enable_label_grouping {
            for label in &grouping_config.grouping_labels {
                let value1 = alert1.labels.get(label);
                let value2 = alert2.labels.get(label);
                if value1 != value2 {
                    return false;
                }
            }
        }

        true
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_alert_error_display() {
        let error = AlertError::ConfigurationError("Test error".to_string());
        assert_eq!(error.to_string(), "Configuration error: Test error");
    }

    #[test]
    fn test_preset_cpu_alert() {
        let alert = presets::cpu_usage_alert(80.0, Duration::from_secs(300));
        assert_eq!(alert.rule_id, "cpu_usage_high");
        assert_eq!(alert.name, "High CPU Usage");
        assert!(matches!(alert.severity, AlertSeverity::High));
    }

    #[test]
    fn test_preset_memory_alert() {
        let alert = presets::memory_usage_alert(90.0, Duration::from_secs(600));
        assert_eq!(alert.rule_id, "memory_usage_high");
        assert_eq!(alert.name, "High Memory Usage");
        assert!(matches!(alert.severity, AlertSeverity::High));
    }

    #[test]
    fn test_utils_validate_alert_rule() {
        let valid_rule = presets::cpu_usage_alert(80.0, Duration::from_secs(300));
        assert!(utils::validate_alert_rule(&valid_rule).is_ok());

        let mut invalid_rule = valid_rule.clone();
        invalid_rule.rule_id = String::new();
        assert!(utils::validate_alert_rule(&invalid_rule).is_err());
    }

    #[test]
    fn test_utils_priority_score() {
        let score_critical = utils::calculate_priority_score(
            &AlertSeverity::Critical,
            Duration::from_secs(3600),
            5.0,
        );
        let score_info = utils::calculate_priority_score(
            &AlertSeverity::Info,
            Duration::from_secs(3600),
            5.0,
        );
        assert!(score_critical > score_info);
    }

    #[test]
    fn test_utils_alert_fingerprint() {
        use std::collections::HashMap;

        let mut labels = HashMap::new();
        labels.insert("instance".to_string(), "web-1".to_string());
        labels.insert("region".to_string(), "us-west".to_string());

        let fingerprint1 = utils::generate_alert_fingerprint("cpu_high", &labels);
        let fingerprint2 = utils::generate_alert_fingerprint("cpu_high", &labels);

        assert_eq!(fingerprint1, fingerprint2);
        assert!(!fingerprint1.is_empty());
    }
}