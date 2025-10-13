//! Quality monitoring and alerting for explanation systems
//!
//! This module provides comprehensive quality monitoring capabilities for explanation systems,
//! including quality metrics calculation, trend detection, alerting, and automated quality assessment.

use crate::{SklResult, SklearsError};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Types of quality metrics that can be monitored
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityMetric {
    // Explanation fidelity metrics
    /// LocalFidelity

    /// LocalFidelity
    LocalFidelity,
    /// GlobalFidelity

    /// GlobalFidelity
    GlobalFidelity,
    /// ModelAgnosticFidelity

    /// ModelAgnosticFidelity
    ModelAgnosticFidelity,

    // Explanation stability metrics
    /// StabilityUnderPerturbation

    /// StabilityUnderPerturbation
    StabilityUnderPerturbation,
    /// ConsistencyAcrossRuns

    /// ConsistencyAcrossRuns
    ConsistencyAcrossRuns,
    /// RobustnessToNoise

    /// RobustnessToNoise
    RobustnessToNoise,

    // Explanation completeness metrics
    /// FeatureCoverage
    FeatureCoverage,
    /// ExplanationCompleteness
    ExplanationCompleteness,
    /// InformationContent
    InformationContent,

    // Performance metrics
    /// ExplanationGenerationTime
    ExplanationGenerationTime,
    /// MemoryUsage
    MemoryUsage,
    /// ComputationalCost
    ComputationalCost,

    // User experience metrics
    /// ExplanationReadability
    ExplanationReadability,
    /// UserSatisfaction
    UserSatisfaction,
    /// InterpretabilityScore
    InterpretabilityScore,

    // System reliability metrics
    /// SystemUptime
    SystemUptime,
    /// ErrorRate
    ErrorRate,
    /// SuccessRate
    SuccessRate,

    // Data quality metrics
    /// DataFreshness
    DataFreshness,
    /// DataCompleteness
    DataCompleteness,
    /// DataConsistency
    DataConsistency,

    // Model quality metrics
    /// ModelAccuracy
    ModelAccuracy,
    /// ModelDrift
    ModelDrift,
    /// ModelPerformance
    ModelPerformance,

    // Compliance metrics
    /// ExplanationCoverage
    ExplanationCoverage,
    /// AuditTrailCompleteness
    AuditTrailCompleteness,
    /// ComplianceScore
    ComplianceScore,

    // Custom metrics
    /// Custom
    Custom(String),
}

impl QualityMetric {
    /// Get the expected range for this metric (0.0 to 1.0 unless otherwise specified)
    pub fn expected_range(&self) -> (f64, f64) {
        match self {
            QualityMetric::ExplanationGenerationTime => (0.0, f64::INFINITY),
            QualityMetric::MemoryUsage => (0.0, f64::INFINITY),
            QualityMetric::ComputationalCost => (0.0, f64::INFINITY),
            _ => (0.0, 1.0),
        }
    }

    /// Check if higher values are better for this metric
    pub fn higher_is_better(&self) -> bool {
        match self {
            QualityMetric::ExplanationGenerationTime
            | QualityMetric::MemoryUsage
            | QualityMetric::ComputationalCost
            | QualityMetric::ErrorRate
            | QualityMetric::ModelDrift => false,
            _ => true,
        }
    }

    /// Get default thresholds for this metric
    pub fn default_thresholds(&self) -> QualityThreshold {
        match self {
            QualityMetric::LocalFidelity
            | QualityMetric::GlobalFidelity
            | QualityMetric::ModelAgnosticFidelity => QualityThreshold::new(0.8, 0.9, 0.95),

            QualityMetric::StabilityUnderPerturbation
            | QualityMetric::ConsistencyAcrossRuns
            | QualityMetric::RobustnessToNoise => QualityThreshold::new(0.7, 0.85, 0.95),

            QualityMetric::FeatureCoverage | QualityMetric::ExplanationCompleteness => {
                QualityThreshold::new(0.8, 0.9, 0.95)
            }

            QualityMetric::ExplanationGenerationTime => {
                QualityThreshold::new(5000.0, 2000.0, 1000.0)
            } // milliseconds

            QualityMetric::SuccessRate | QualityMetric::SystemUptime => {
                QualityThreshold::new(0.95, 0.98, 0.99)
            }

            QualityMetric::ErrorRate => QualityThreshold::new(0.05, 0.02, 0.01),

            QualityMetric::ModelAccuracy | QualityMetric::ModelPerformance => {
                QualityThreshold::new(0.85, 0.9, 0.95)
            }

            _ => QualityThreshold::new(0.7, 0.8, 0.9),
        }
    }
}

impl std::fmt::Display for QualityMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualityMetric::LocalFidelity => write!(f, "local_fidelity"),
            QualityMetric::GlobalFidelity => write!(f, "global_fidelity"),
            QualityMetric::ModelAgnosticFidelity => write!(f, "model_agnostic_fidelity"),
            QualityMetric::StabilityUnderPerturbation => write!(f, "stability_under_perturbation"),
            QualityMetric::ConsistencyAcrossRuns => write!(f, "consistency_across_runs"),
            QualityMetric::RobustnessToNoise => write!(f, "robustness_to_noise"),
            QualityMetric::FeatureCoverage => write!(f, "feature_coverage"),
            QualityMetric::ExplanationCompleteness => write!(f, "explanation_completeness"),
            QualityMetric::InformationContent => write!(f, "information_content"),
            QualityMetric::ExplanationGenerationTime => write!(f, "explanation_generation_time"),
            QualityMetric::MemoryUsage => write!(f, "memory_usage"),
            QualityMetric::ComputationalCost => write!(f, "computational_cost"),
            QualityMetric::ExplanationReadability => write!(f, "explanation_readability"),
            QualityMetric::UserSatisfaction => write!(f, "user_satisfaction"),
            QualityMetric::InterpretabilityScore => write!(f, "interpretability_score"),
            QualityMetric::SystemUptime => write!(f, "system_uptime"),
            QualityMetric::ErrorRate => write!(f, "error_rate"),
            QualityMetric::SuccessRate => write!(f, "success_rate"),
            QualityMetric::DataFreshness => write!(f, "data_freshness"),
            QualityMetric::DataCompleteness => write!(f, "data_completeness"),
            QualityMetric::DataConsistency => write!(f, "data_consistency"),
            QualityMetric::ModelAccuracy => write!(f, "model_accuracy"),
            QualityMetric::ModelDrift => write!(f, "model_drift"),
            QualityMetric::ModelPerformance => write!(f, "model_performance"),
            QualityMetric::ExplanationCoverage => write!(f, "explanation_coverage"),
            QualityMetric::AuditTrailCompleteness => write!(f, "audit_trail_completeness"),
            QualityMetric::ComplianceScore => write!(f, "compliance_score"),
            QualityMetric::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// Quality thresholds for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThreshold {
    /// Warning threshold
    pub warning: f64,
    /// Critical threshold
    pub critical: f64,
    /// Excellent threshold
    pub excellent: f64,
}

impl QualityThreshold {
    /// Create new quality threshold
    pub fn new(warning: f64, critical: f64, excellent: f64) -> Self {
        Self {
            warning,
            critical,
            excellent,
        }
    }

    /// Evaluate the quality level for a given value
    pub fn evaluate(&self, value: f64, higher_is_better: bool) -> QualityLevel {
        if higher_is_better {
            if value >= self.excellent {
                QualityLevel::Excellent
            } else if value >= self.critical {
                QualityLevel::Good
            } else if value >= self.warning {
                QualityLevel::Warning
            } else {
                QualityLevel::Critical
            }
        } else {
            if value <= self.excellent {
                QualityLevel::Excellent
            } else if value <= self.critical {
                QualityLevel::Good
            } else if value <= self.warning {
                QualityLevel::Warning
            } else {
                QualityLevel::Critical
            }
        }
    }
}

/// Quality levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Excellent

    /// Excellent
    Excellent,
    /// Good

    /// Good
    Good,
    /// Warning

    /// Warning
    Warning,
    /// Critical

    /// Critical
    Critical,
}

impl QualityLevel {
    /// Get numeric score for the quality level
    pub fn score(&self) -> u8 {
        match self {
            QualityLevel::Excellent => 4,
            QualityLevel::Good => 3,
            QualityLevel::Warning => 2,
            QualityLevel::Critical => 1,
        }
    }

    /// Check if this quality level requires immediate attention
    pub fn requires_attention(&self) -> bool {
        matches!(self, QualityLevel::Critical | QualityLevel::Warning)
    }
}

impl std::fmt::Display for QualityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualityLevel::Excellent => write!(f, "excellent"),
            QualityLevel::Good => write!(f, "good"),
            QualityLevel::Warning => write!(f, "warning"),
            QualityLevel::Critical => write!(f, "critical"),
        }
    }
}

/// Direction of quality trend
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTrendDirection {
    /// Improving

    /// Improving
    Improving,
    /// Stable

    /// Stable
    Stable,
    /// Degrading

    /// Degrading
    Degrading,
    /// Unknown

    /// Unknown
    Unknown,
}

/// Quality trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrend {
    /// Direction of the trend
    pub direction: QualityTrendDirection,
    /// Trend slope (rate of change per unit time)
    pub slope: f64,
    /// Confidence in the trend (0.0 to 1.0)
    pub confidence: f64,
    /// Time period over which the trend was calculated
    pub period: Duration,
}

impl QualityTrend {
    /// Create a new quality trend
    pub fn new(
        direction: QualityTrendDirection,
        slope: f64,
        confidence: f64,
        period: Duration,
    ) -> Self {
        Self {
            direction,
            slope,
            confidence,
            period,
        }
    }

    /// Determine if the trend is significant
    pub fn is_significant(&self, min_confidence: f64) -> bool {
        self.confidence >= min_confidence && self.direction != QualityTrendDirection::Unknown
    }
}

/// Quality alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAlert {
    /// Unique alert ID
    pub id: String,
    /// Metric that triggered the alert
    pub metric: QualityMetric,
    /// Current quality level
    pub quality_level: QualityLevel,
    /// Current metric value
    pub current_value: f64,
    /// Threshold that was violated
    pub threshold_violated: String,
    /// When the alert was triggered
    pub triggered_at: DateTime<Utc>,
    /// Alert message
    pub message: String,
    /// Additional context
    pub context: HashMap<String, String>,
    /// Whether the alert has been acknowledged
    pub acknowledged: bool,
    /// When the alert was acknowledged
    pub acknowledged_at: Option<DateTime<Utc>>,
    /// Who acknowledged the alert
    pub acknowledged_by: Option<String>,
}

impl QualityAlert {
    /// Create a new quality alert
    pub fn new(
        metric: QualityMetric,
        quality_level: QualityLevel,
        current_value: f64,
        threshold_violated: String,
        message: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            metric,
            quality_level,
            current_value,
            threshold_violated,
            triggered_at: Utc::now(),
            message,
            context: HashMap::new(),
            acknowledged: false,
            acknowledged_at: None,
            acknowledged_by: None,
        }
    }

    /// Add context to the alert
    pub fn with_context(mut self, key: String, value: String) -> Self {
        self.context.insert(key, value);
        self
    }

    /// Acknowledge the alert
    pub fn acknowledge(&mut self, user: String) {
        self.acknowledged = true;
        self.acknowledged_at = Some(Utc::now());
        self.acknowledged_by = Some(user);
    }

    /// Check if the alert is critical
    pub fn is_critical(&self) -> bool {
        self.quality_level == QualityLevel::Critical
    }
}

/// Rule for triggering quality alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Metric to monitor
    pub metric: QualityMetric,
    /// Quality thresholds
    pub thresholds: QualityThreshold,
    /// Whether the rule is enabled
    pub enabled: bool,
    /// Minimum samples required before alerting
    pub min_samples: usize,
    /// Time window for evaluation
    pub evaluation_window: Duration,
    /// Cooldown period between alerts
    pub cooldown_period: Duration,
    /// Last time this rule triggered an alert
    pub last_alert: Option<DateTime<Utc>>,
}

impl AlertRule {
    /// Create a new alert rule
    pub fn new(name: String, metric: QualityMetric, thresholds: QualityThreshold) -> Self {
        Self {
            name,
            metric,
            thresholds,
            enabled: true,
            min_samples: 5,
            evaluation_window: Duration::minutes(10),
            cooldown_period: Duration::minutes(5),
            last_alert: None,
        }
    }

    /// Check if the rule should be evaluated (not in cooldown)
    pub fn can_evaluate(&self) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(last_alert) = self.last_alert {
            Utc::now() - last_alert >= self.cooldown_period
        } else {
            true
        }
    }

    /// Evaluate the rule with current data points
    pub fn evaluate(&self, data_points: &[(DateTime<Utc>, f64)]) -> Option<QualityAlert> {
        if !self.can_evaluate() || data_points.len() < self.min_samples {
            return None;
        }

        // Filter data points within evaluation window
        let cutoff_time = Utc::now() - self.evaluation_window;
        let recent_points: Vec<_> = data_points
            .iter()
            .filter(|(timestamp, _)| *timestamp >= cutoff_time)
            .collect();

        if recent_points.len() < self.min_samples {
            return None;
        }

        // Calculate average value in the window
        let avg_value =
            recent_points.iter().map(|(_, value)| value).sum::<f64>() / recent_points.len() as f64;

        // Evaluate quality level
        let quality_level = self
            .thresholds
            .evaluate(avg_value, self.metric.higher_is_better());

        // Check if alert should be triggered
        if quality_level.requires_attention() {
            let threshold_violated = match quality_level {
                QualityLevel::Warning => "warning",
                QualityLevel::Critical => "critical",
                _ => return None,
            };

            let message = format!(
                "Quality metric '{}' has reached {} level with value {:.3}",
                self.metric, quality_level, avg_value
            );

            Some(
                QualityAlert::new(
                    self.metric.clone(),
                    quality_level,
                    avg_value,
                    threshold_violated.to_string(),
                    message,
                )
                .with_context("rule_name".to_string(), self.name.clone())
                .with_context("sample_count".to_string(), recent_points.len().to_string()),
            )
        } else {
            None
        }
    }

    /// Update the last alert time
    pub fn update_last_alert(&mut self) {
        self.last_alert = Some(Utc::now());
    }
}

/// Configuration for quality monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMonitorConfig {
    /// Maximum number of data points to keep per metric
    pub max_data_points: usize,
    /// Interval between quality evaluations
    pub evaluation_interval: Duration,
    /// Whether to enable automatic alerting
    pub enable_alerting: bool,
    /// Default retention period for quality data
    pub data_retention_period: Duration,
    /// Minimum confidence for trend detection
    pub trend_confidence_threshold: f64,
}

impl Default for QualityMonitorConfig {
    fn default() -> Self {
        Self {
            max_data_points: 1000,
            evaluation_interval: Duration::minutes(1),
            enable_alerting: true,
            data_retention_period: Duration::days(30),
            trend_confidence_threshold: 0.7,
        }
    }
}

/// Quality data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDataPoint {
    /// When the measurement was taken
    pub timestamp: DateTime<Utc>,
    /// Metric value
    pub value: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl QualityDataPoint {
    /// Create a new quality data point
    pub fn new(value: f64) -> Self {
        Self {
            timestamp: Utc::now(),
            value,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Time series data for a quality metric
#[derive(Debug)]
pub struct QualityTimeSeries {
    /// The metric being tracked
    metric: QualityMetric,
    /// Historical data points
    data_points: VecDeque<QualityDataPoint>,
    /// Maximum number of data points to retain
    max_data_points: usize,
    /// Current trend information
    current_trend: Option<QualityTrend>,
}

impl QualityTimeSeries {
    /// Create a new quality time series
    pub fn new(metric: QualityMetric, max_data_points: usize) -> Self {
        Self {
            metric,
            data_points: VecDeque::new(),
            max_data_points,
            current_trend: None,
        }
    }

    /// Add a data point
    pub fn add_data_point(&mut self, data_point: QualityDataPoint) {
        self.data_points.push_back(data_point);

        // Maintain maximum size
        while self.data_points.len() > self.max_data_points {
            self.data_points.pop_front();
        }

        // Update trend if we have enough data
        if self.data_points.len() >= 10 {
            self.update_trend();
        }
    }

    /// Get recent data points
    pub fn get_recent_data(&self, duration: Duration) -> Vec<(DateTime<Utc>, f64)> {
        let cutoff_time = Utc::now() - duration;
        self.data_points
            .iter()
            .filter(|dp| dp.timestamp >= cutoff_time)
            .map(|dp| (dp.timestamp, dp.value))
            .collect()
    }

    /// Get the current value (most recent data point)
    pub fn current_value(&self) -> Option<f64> {
        self.data_points.back().map(|dp| dp.value)
    }

    /// Get the current trend
    pub fn current_trend(&self) -> Option<&QualityTrend> {
        self.current_trend.as_ref()
    }

    /// Calculate statistics for recent data
    pub fn calculate_statistics(&self, duration: Duration) -> Option<QualityStatistics> {
        let recent_data = self.get_recent_data(duration);
        if recent_data.is_empty() {
            return None;
        }

        let values: Vec<f64> = recent_data.iter().map(|(_, v)| *v).collect();
        let count = values.len();
        let sum = values.iter().sum::<f64>();
        let mean = sum / count as f64;

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_values[0];
        let max = sorted_values[count - 1];
        let median = if count % 2 == 0 {
            (sorted_values[count / 2 - 1] + sorted_values[count / 2]) / 2.0
        } else {
            sorted_values[count / 2]
        };

        Some(QualityStatistics {
            count,
            mean,
            median,
            std_dev,
            min,
            max,
            trend: self.current_trend.clone(),
        })
    }

    fn update_trend(&mut self) {
        let min_points = 10;
        if self.data_points.len() < min_points {
            return;
        }

        // Use last 20 points for trend calculation
        let num_points = std::cmp::min(20, self.data_points.len());
        let points: Vec<_> = self
            .data_points
            .iter()
            .rev()
            .take(num_points)
            .map(|dp| {
                let timestamp_secs = dp.timestamp.timestamp() as f64;
                (timestamp_secs, dp.value)
            })
            .collect();

        if let Some(trend) = calculate_trend(&points) {
            self.current_trend = Some(trend);
        }
    }
}

/// Quality statistics for a time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStatistics {
    /// Number of data points
    pub count: usize,
    /// Average value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Current trend
    pub trend: Option<QualityTrend>,
}

/// Main quality monitoring system
#[derive(Debug)]
pub struct ExplanationQualityMonitor {
    /// Configuration
    config: QualityMonitorConfig,
    /// Time series data for each metric
    metrics: HashMap<QualityMetric, QualityTimeSeries>,
    /// Alert rules
    alert_rules: HashMap<String, AlertRule>,
    /// Active alerts
    active_alerts: Vec<QualityAlert>,
    /// Alert history
    alert_history: VecDeque<QualityAlert>,
}

impl ExplanationQualityMonitor {
    /// Create a new quality monitor
    pub fn new(config: QualityMonitorConfig) -> Self {
        let mut monitor = Self {
            config,
            metrics: HashMap::new(),
            alert_rules: HashMap::new(),
            active_alerts: Vec::new(),
            alert_history: VecDeque::new(),
        };

        // Add default alert rules
        monitor.add_default_alert_rules();
        monitor
    }

    /// Record a quality measurement
    pub fn record_measurement(&mut self, metric: QualityMetric, value: f64) -> SklResult<()> {
        self.record_measurement_with_metadata(metric, value, HashMap::new())
    }

    /// Record a quality measurement with metadata
    pub fn record_measurement_with_metadata(
        &mut self,
        metric: QualityMetric,
        value: f64,
        metadata: HashMap<String, String>,
    ) -> SklResult<()> {
        // Validate metric value
        let (min, max) = metric.expected_range();
        if value < min || value > max {
            return Err(SklearsError::InvalidParameter {
                name: "metric_value".to_string(),
                reason: format!(
                    "Value {:.3} for metric '{}' is outside expected range [{:.3}, {:.3}]",
                    value, metric, min, max
                ),
            });
        }

        // Get or create time series for this metric
        let time_series = self
            .metrics
            .entry(metric.clone())
            .or_insert_with(|| QualityTimeSeries::new(metric.clone(), self.config.max_data_points));

        // Add data point
        let mut data_point = QualityDataPoint::new(value);
        for (key, value) in metadata {
            data_point = data_point.with_metadata(key, value);
        }

        time_series.add_data_point(data_point);

        // Evaluate alert rules if enabled
        if self.config.enable_alerting {
            self.evaluate_alert_rules_for_metric(&metric);
        }

        Ok(())
    }

    /// Add an alert rule
    pub fn add_alert_rule(&mut self, rule: AlertRule) {
        self.alert_rules.insert(rule.name.clone(), rule);
    }

    /// Remove an alert rule
    pub fn remove_alert_rule(&mut self, rule_name: &str) {
        self.alert_rules.remove(rule_name);
    }

    /// Get quality statistics for a metric
    pub fn get_statistics(
        &self,
        metric: &QualityMetric,
        duration: Duration,
    ) -> Option<QualityStatistics> {
        self.metrics.get(metric)?.calculate_statistics(duration)
    }

    /// Get current value for a metric
    pub fn get_current_value(&self, metric: &QualityMetric) -> Option<f64> {
        self.metrics.get(metric)?.current_value()
    }

    /// Get current trend for a metric
    pub fn get_current_trend(&self, metric: &QualityMetric) -> Option<&QualityTrend> {
        self.metrics.get(metric)?.current_trend()
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> &Vec<QualityAlert> {
        &self.active_alerts
    }

    /// Get critical alerts
    pub fn get_critical_alerts(&self) -> Vec<&QualityAlert> {
        self.active_alerts
            .iter()
            .filter(|alert| alert.is_critical())
            .collect()
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&mut self, alert_id: &str, user: String) -> SklResult<()> {
        if let Some(alert) = self.active_alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.acknowledge(user);
            Ok(())
        } else {
            Err(SklearsError::InvalidParameter {
                name: "alert_id".to_string(),
                reason: format!("Alert '{}' not found", alert_id),
            })
        }
    }

    /// Clear resolved alerts
    pub fn clear_resolved_alerts(&mut self) {
        // Move acknowledged alerts to history
        let mut resolved_alerts = Vec::new();
        self.active_alerts.retain(|alert| {
            if alert.acknowledged {
                resolved_alerts.push(alert.clone());
                false
            } else {
                true
            }
        });

        // Add to history
        for alert in resolved_alerts {
            self.alert_history.push_back(alert);

            // Maintain history size
            while self.alert_history.len() > 1000 {
                self.alert_history.pop_front();
            }
        }
    }

    /// Get overall system quality score
    pub fn get_overall_quality_score(&self) -> f64 {
        if self.metrics.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self
            .metrics
            .iter()
            .filter_map(|(metric, time_series)| {
                time_series.current_value().map(|value| {
                    let threshold = metric.default_thresholds();
                    let quality_level = threshold.evaluate(value, metric.higher_is_better());
                    quality_level.score() as f64
                })
            })
            .sum();

        let metric_count = self.metrics.len() as f64;
        (total_score / metric_count) / 4.0 // Normalize to 0-1 scale
    }

    fn add_default_alert_rules(&mut self) {
        let default_metrics = vec![
            QualityMetric::LocalFidelity,
            QualityMetric::GlobalFidelity,
            QualityMetric::StabilityUnderPerturbation,
            QualityMetric::ExplanationGenerationTime,
            QualityMetric::SuccessRate,
            QualityMetric::ErrorRate,
        ];

        for metric in default_metrics {
            let rule_name = format!("default_{}", metric);
            let thresholds = metric.default_thresholds();
            let rule = AlertRule::new(rule_name, metric, thresholds);
            self.add_alert_rule(rule);
        }
    }

    fn evaluate_alert_rules_for_metric(&mut self, metric: &QualityMetric) {
        let rules_to_evaluate: Vec<_> = self
            .alert_rules
            .values()
            .filter(|rule| &rule.metric == metric)
            .cloned()
            .collect();

        if let Some(time_series) = self.metrics.get(metric) {
            let recent_data = time_series.get_recent_data(Duration::hours(1));

            for mut rule in rules_to_evaluate {
                if let Some(alert) = rule.evaluate(&recent_data) {
                    rule.update_last_alert();
                    self.alert_rules.insert(rule.name.clone(), rule);
                    self.active_alerts.push(alert);
                }
            }
        }
    }
}

/// Calculate trend from time series data points
fn calculate_trend(points: &[(f64, f64)]) -> Option<QualityTrend> {
    if points.len() < 5 {
        return None;
    }

    // Calculate linear regression
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_x_squared: f64 = points.iter().map(|(x, _)| x * x).sum();

    let denominator = n * sum_x_squared - sum_x * sum_x;
    if denominator.abs() < f64::EPSILON {
        return Some(QualityTrend::new(
            QualityTrendDirection::Stable,
            0.0,
            0.0,
            Duration::zero(),
        ));
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;

    // Calculate correlation coefficient (confidence)
    let sum_y_squared: f64 = points.iter().map(|(_, y)| y * y).sum();
    let numerator = n * sum_xy - sum_x * sum_y;
    let r_squared = (numerator * numerator)
        / ((n * sum_x_squared - sum_x * sum_x) * (n * sum_y_squared - sum_y * sum_y));
    let confidence = r_squared.sqrt();

    // Determine trend direction
    let direction = if slope.abs() < 1e-6 {
        QualityTrendDirection::Stable
    } else if slope > 0.0 {
        QualityTrendDirection::Improving
    } else {
        QualityTrendDirection::Degrading
    };

    // Calculate time period
    let first_time = points.first().unwrap().0 as i64;
    let last_time = points.last().unwrap().0 as i64;
    let period = Duration::seconds(last_time - first_time);

    Some(QualityTrend::new(direction, slope, confidence, period))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_metric_display() {
        assert_eq!(QualityMetric::LocalFidelity.to_string(), "local_fidelity");
        assert_eq!(
            QualityMetric::Custom("test".to_string()).to_string(),
            "custom_test"
        );
    }

    #[test]
    fn test_quality_threshold() {
        let threshold = QualityThreshold::new(0.7, 0.8, 0.9);

        assert_eq!(threshold.evaluate(0.95, true), QualityLevel::Excellent);
        assert_eq!(threshold.evaluate(0.85, true), QualityLevel::Good);
        assert_eq!(threshold.evaluate(0.75, true), QualityLevel::Warning);
        assert_eq!(threshold.evaluate(0.65, true), QualityLevel::Critical);

        // Test lower is better
        assert_eq!(threshold.evaluate(0.65, false), QualityLevel::Excellent);
        assert_eq!(threshold.evaluate(0.95, false), QualityLevel::Critical);
    }

    #[test]
    fn test_quality_alert() {
        let mut alert = QualityAlert::new(
            QualityMetric::LocalFidelity,
            QualityLevel::Warning,
            0.65,
            "warning".to_string(),
            "Test alert".to_string(),
        );

        assert!(!alert.acknowledged);
        assert!(!alert.is_critical());

        alert.acknowledge("user1".to_string());
        assert!(alert.acknowledged);
        assert_eq!(alert.acknowledged_by, Some("user1".to_string()));
    }

    #[test]
    fn test_alert_rule() {
        let threshold = QualityThreshold::new(0.7, 0.8, 0.9);
        let rule = AlertRule::new(
            "test_rule".to_string(),
            QualityMetric::LocalFidelity,
            threshold,
        );

        assert!(rule.can_evaluate());

        // Create data points that should trigger an alert
        let now = Utc::now();
        let data_points = vec![
            (now - Duration::minutes(5), 0.6), // Critical level
            (now - Duration::minutes(4), 0.65),
            (now - Duration::minutes(3), 0.62),
            (now - Duration::minutes(2), 0.61),
            (now - Duration::minutes(1), 0.63),
        ];

        let alert = rule.evaluate(&data_points);
        assert!(alert.is_some());

        let alert = alert.unwrap();
        assert_eq!(alert.quality_level, QualityLevel::Critical);
    }

    #[test]
    fn test_quality_time_series() {
        let mut time_series = QualityTimeSeries::new(QualityMetric::LocalFidelity, 100);

        // Add some data points
        for i in 0..10 {
            let value = 0.8 + (i as f64) * 0.01; // Improving trend
            time_series.add_data_point(QualityDataPoint::new(value));
        }

        assert_eq!(time_series.current_value(), Some(0.89));

        let recent_data = time_series.get_recent_data(Duration::hours(1));
        assert_eq!(recent_data.len(), 10);

        let stats = time_series.calculate_statistics(Duration::hours(1));
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, 0.8);
        assert_eq!(stats.max, 0.89);
    }

    #[test]
    fn test_quality_monitor() {
        let config = QualityMonitorConfig::default();
        let mut monitor = ExplanationQualityMonitor::new(config);

        // Record some measurements
        monitor
            .record_measurement(QualityMetric::LocalFidelity, 0.85)
            .unwrap();
        monitor
            .record_measurement(QualityMetric::LocalFidelity, 0.87)
            .unwrap();
        monitor
            .record_measurement(QualityMetric::LocalFidelity, 0.82)
            .unwrap();

        assert_eq!(
            monitor.get_current_value(&QualityMetric::LocalFidelity),
            Some(0.82)
        );

        let stats = monitor.get_statistics(&QualityMetric::LocalFidelity, Duration::hours(1));
        assert!(stats.is_some());

        let overall_score = monitor.get_overall_quality_score();
        assert!(overall_score > 0.0 && overall_score <= 1.0);
    }

    #[test]
    fn test_trend_calculation() {
        // Test improving trend
        let points = vec![(0.0, 0.7), (1.0, 0.75), (2.0, 0.8), (3.0, 0.85), (4.0, 0.9)];

        let trend = calculate_trend(&points).unwrap();
        assert_eq!(trend.direction, QualityTrendDirection::Improving);
        assert!(trend.slope > 0.0);
        assert!(trend.confidence > 0.9); // Strong correlation

        // Test degrading trend
        let points = vec![(0.0, 0.9), (1.0, 0.85), (2.0, 0.8), (3.0, 0.75), (4.0, 0.7)];

        let trend = calculate_trend(&points).unwrap();
        assert_eq!(trend.direction, QualityTrendDirection::Degrading);
        assert!(trend.slope < 0.0);
    }

    #[test]
    fn test_invalid_measurement() {
        let config = QualityMonitorConfig::default();
        let mut monitor = ExplanationQualityMonitor::new(config);

        // Try to record invalid value (outside expected range)
        let result = monitor.record_measurement(QualityMetric::LocalFidelity, 1.5);
        assert!(result.is_err());
    }
}
