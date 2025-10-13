//! Shared types and configuration structures for alert management
//!
//! This module provides common types, enums, and configuration structures
//! used throughout the alert management system including severity levels,
//! threshold operators, and shared configuration types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Alert severity levels with extended classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Critical alerts requiring immediate action
    Critical,
    /// High priority alerts requiring urgent attention
    High,
    /// Medium priority alerts requiring attention within business hours
    Medium,
    /// Warning alerts for potential issues
    Warning,
    /// Informational alerts for monitoring
    Info,
    /// Low priority alerts for background monitoring
    Low,
    /// Debug level alerts for development
    Debug,
}

impl Default for AlertSeverity {
    fn default() -> Self {
        Self::Medium
    }
}

/// Alert threshold configuration with advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    /// Comparison operator
    pub operator: ThresholdOperator,
    /// Threshold value
    pub value: f64,
    /// Optional hysteresis for threshold stability
    pub hysteresis: Option<f64>,
    /// Optional percentage-based threshold
    pub percentage: Option<f64>,
    /// Dynamic threshold configuration
    pub dynamic_threshold: Option<DynamicThreshold>,
    /// Multi-dimensional thresholds
    pub multi_dimensional: Vec<MultiDimensionalThreshold>,
    /// Threshold adjustment rules
    pub adjustment_rules: Vec<ThresholdAdjustmentRule>,
}

impl Default for AlertThreshold {
    fn default() -> Self {
        Self {
            operator: ThresholdOperator::GreaterThan,
            value: 0.0,
            hysteresis: None,
            percentage: None,
            dynamic_threshold: None,
            multi_dimensional: Vec::new(),
            adjustment_rules: Vec::new(),
        }
    }
}

/// Dynamic threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicThreshold {
    /// Statistical method for dynamic calculation
    pub method: DynamicMethod,
    /// Window size for calculation
    pub window_size: Duration,
    /// Number of standard deviations for statistical methods
    pub std_dev_multiplier: f64,
    /// Minimum number of samples required
    pub min_samples: usize,
    /// Update frequency
    pub update_frequency: Duration,
}

/// Dynamic threshold calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DynamicMethod {
    MovingAverage,
    ExponentialSmoothing,
    SeasonalDecomposition,
    ARIMA,
    MachineLearning,
}

/// Multi-dimensional threshold for complex metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDimensionalThreshold {
    /// Metric dimensions
    pub dimensions: Vec<String>,
    /// Threshold values for each dimension
    pub values: Vec<f64>,
    /// Operators for each dimension
    pub operators: Vec<ThresholdOperator>,
    /// Combination logic
    pub combination_logic: CombinationLogic,
}

/// Logic for combining multi-dimensional thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CombinationLogic {
    And,
    Or,
    Weighted { weights: Vec<f64> },
    Majority,
    AtLeast { count: usize },
}

/// Threshold adjustment rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdAdjustmentRule {
    /// Condition for applying the rule
    pub condition: AdjustmentCondition,
    /// Type of adjustment
    pub adjustment_type: AdjustmentType,
    /// Adjustment value
    pub adjustment_value: f64,
    /// Rule priority
    pub priority: i32,
}

/// Conditions for threshold adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdjustmentCondition {
    TimeOfDay { start_hour: u8, end_hour: u8 },
    DayOfWeek { days: Vec<Weekday> },
    Load { cpu_threshold: f64, memory_threshold: f64 },
    NetworkCondition { latency_threshold: f64 },
    Custom { expression: String },
}

/// Days of the week for adjustment conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Weekday {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

/// Types of threshold adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdjustmentType {
    Absolute,
    Percentage,
    Multiplier,
    AddConstant,
    Dynamic,
}

/// Threshold comparison operators with extended functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdOperator {
    /// Greater than comparison
    GreaterThan,
    /// Greater than or equal comparison
    GreaterThanOrEqual,
    /// Less than comparison
    LessThan,
    /// Less than or equal comparison
    LessThanOrEqual,
    /// Equality comparison
    Equal,
    /// Not equal comparison
    NotEqual,
    /// Within range comparison
    InRange { min: f64, max: f64 },
    /// Outside range comparison
    OutOfRange { min: f64, max: f64 },
    /// Change rate comparison
    ChangeRate { rate: f64, window: Duration },
    /// Trend analysis
    Trend { direction: TrendDirection, confidence: f64 },
    /// Statistical comparison
    Statistical { method: StatisticalComparison },
    /// Regular expression for string values
    Regex { pattern: String },
    /// Custom comparison logic
    Custom { expression: String },
}

impl Default for ThresholdOperator {
    fn default() -> Self {
        Self::GreaterThan
    }
}

/// Trend direction for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Statistical comparison methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalComparison {
    ZScore { threshold: f64 },
    Percentile { percentile: f64 },
    InterquartileRange { multiplier: f64 },
    MahalanobisDistance { threshold: f64 },
}

/// Evaluation priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationPriority {
    Realtime,
    High,
    Normal,
    Low,
    Background,
}

impl Default for EvaluationPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Resource limits for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResourceLimits {
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f64,
    /// Maximum memory usage in bytes
    pub max_memory_usage: usize,
    /// Maximum evaluation time
    pub max_evaluation_time: Duration,
    /// Maximum concurrent evaluations
    pub max_concurrent_evaluations: usize,
}

impl Default for EvaluationResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_usage: 50.0,
            max_memory_usage: 100 * 1024 * 1024, // 100MB
            max_evaluation_time: Duration::from_secs(30),
            max_concurrent_evaluations: 10,
        }
    }
}

/// Fallback strategies for evaluation failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStrategy {
    LastKnownGood,
    DefaultValue { value: f64 },
    Skip,
    Retry { max_retries: usize, delay: Duration },
    AlternativeMethod { method: String },
}

impl Default for FallbackStrategy {
    fn default() -> Self {
        Self::LastKnownGood
    }
}

/// Quality of service settings for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationQoS {
    /// Maximum evaluation latency
    pub max_latency: Duration,
    /// Evaluation priority
    pub priority: EvaluationPriority,
    /// Resource limits
    pub resource_limits: EvaluationResourceLimits,
    /// Fallback strategy
    pub fallback_strategy: FallbackStrategy,
}

impl Default for EvaluationQoS {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_secs(10),
            priority: EvaluationPriority::Normal,
            resource_limits: EvaluationResourceLimits::default(),
            fallback_strategy: FallbackStrategy::LastKnownGood,
        }
    }
}

/// Alert routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRouting {
    /// Default notification channels
    pub default_channels: Vec<String>,
    /// Severity-based routing
    pub severity_routing: HashMap<AlertSeverity, Vec<String>>,
    /// Label-based routing rules
    pub label_routing: Vec<RoutingRule>,
    /// Escalation rules
    pub escalation_rules: Vec<EscalationRule>,
}

impl Default for AlertRouting {
    fn default() -> Self {
        Self {
            default_channels: vec!["default".to_string()],
            severity_routing: HashMap::new(),
            label_routing: Vec::new(),
            escalation_rules: Vec::new(),
        }
    }
}

/// Routing rule based on labels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Label selector
    pub label_selector: HashMap<String, String>,
    /// Target channels
    pub channels: Vec<String>,
    /// Rule priority
    pub priority: i32,
}

/// Escalation rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    /// Condition for escalation
    pub condition: EscalationCondition,
    /// Target channels for escalation
    pub escalation_channels: Vec<String>,
    /// Delay before escalation
    pub escalation_delay: Duration,
    /// Maximum escalation attempts
    pub max_escalations: usize,
}

/// Conditions that trigger escalation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationCondition {
    UnacknowledgedDuration(Duration),
    UnresolvedDuration(Duration),
    RepeatCount(usize),
    SeverityIncrease,
    CustomCondition(String),
}

/// Performance snapshot for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: SystemTime,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Additional context
    pub context: HashMap<String, String>,
}

impl Default for PerformanceSnapshot {
    fn default() -> Self {
        Self {
            timestamp: SystemTime::now(),
            metrics: HashMap::new(),
            context: HashMap::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_severity_default() {
        let severity = AlertSeverity::default();
        assert_eq!(severity, AlertSeverity::Medium);
    }

    #[test]
    fn test_threshold_operator_default() {
        let operator = ThresholdOperator::default();
        match operator {
            ThresholdOperator::GreaterThan => (),
            _ => panic!("Expected GreaterThan as default"),
        }
    }

    #[test]
    fn test_alert_threshold_default() {
        let threshold = AlertThreshold::default();
        assert_eq!(threshold.value, 0.0);
        assert!(threshold.hysteresis.is_none());
        assert!(threshold.percentage.is_none());
    }

    #[test]
    fn test_evaluation_qos_default() {
        let qos = EvaluationQoS::default();
        assert_eq!(qos.max_latency, Duration::from_secs(10));
        match qos.priority {
            EvaluationPriority::Normal => (),
            _ => panic!("Expected Normal as default priority"),
        }
    }

    #[test]
    fn test_alert_routing_default() {
        let routing = AlertRouting::default();
        assert_eq!(routing.default_channels, vec!["default".to_string()]);
        assert!(routing.severity_routing.is_empty());
        assert!(routing.label_routing.is_empty());
    }

    #[test]
    fn test_threshold_operator_variants() {
        let operators = vec![
            ThresholdOperator::GreaterThan,
            ThresholdOperator::LessThan,
            ThresholdOperator::Equal,
            ThresholdOperator::InRange { min: 0.0, max: 100.0 },
        ];

        for operator in operators {
            // Ensure all variants can be created
            match operator {
                ThresholdOperator::GreaterThan => (),
                ThresholdOperator::LessThan => (),
                ThresholdOperator::Equal => (),
                ThresholdOperator::InRange { min, max } => {
                    assert!(min <= max);
                }
                _ => (),
            }
        }
    }

    #[test]
    fn test_dynamic_threshold_configuration() {
        let dynamic_threshold = DynamicThreshold {
            method: DynamicMethod::MovingAverage,
            window_size: Duration::from_secs(3600),
            std_dev_multiplier: 2.0,
            min_samples: 10,
            update_frequency: Duration::from_secs(300),
        };

        assert!(dynamic_threshold.std_dev_multiplier > 0.0);
        assert!(dynamic_threshold.min_samples > 0);
    }

    #[test]
    fn test_multi_dimensional_threshold() {
        let threshold = MultiDimensionalThreshold {
            dimensions: vec!["cpu".to_string(), "memory".to_string()],
            values: vec![80.0, 90.0],
            operators: vec![ThresholdOperator::GreaterThan, ThresholdOperator::GreaterThan],
            combination_logic: CombinationLogic::And,
        };

        assert_eq!(threshold.dimensions.len(), threshold.values.len());
        assert_eq!(threshold.values.len(), threshold.operators.len());
    }

    #[test]
    fn test_performance_snapshot() {
        let snapshot = PerformanceSnapshot::default();
        assert!(snapshot.metrics.is_empty());
        assert!(snapshot.context.is_empty());
    }
}