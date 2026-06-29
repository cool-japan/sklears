//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::comparison_engine::BaselineManager;
use super::super::config_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::types::{
    AlertSuppression, BaselineComparisonMethod, ChangeType, DetectionSensitivity,
    DetectionStatistics, EffectSizeAnalysis, EscalationPolicy, ImplementationCost, MetricTrend,
    NotificationChannel, PotentialCause, RateLimiting, RegressionAlertRule, RegressionCondition,
    RegressionRecommendationType, RegressionReport, RemediationPriority, SignificanceTesting,
    SuppressionCondition, SystemHealthScore, ThresholdHistoryEntry, ThresholdType, TrendPrediction,
    TrendSummary, WarningSeverity, WarningType,
};

/// User impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserImpactAssessment {
    pub affected_users: usize,
    pub user_experience_degradation: f64,
    pub feature_availability: f64,
    pub performance_perception: f64,
}
/// Regression cache for performance optimization
#[derive(Debug)]
pub struct RegressionCache {
    pub(super) cache: HashMap<String, RegressionReport>,
    pub(super) max_size: usize,
    pub(super) hit_count: u64,
    pub(super) miss_count: u64,
}
impl RegressionCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_size: 1000,
            hit_count: 0,
            miss_count: 0,
        }
    }
    pub fn get(&mut self, key: &str) -> Option<&RegressionReport> {
        if let Some(report) = self.cache.get(key) {
            self.hit_count += 1;
            Some(report)
        } else {
            self.miss_count += 1;
            None
        }
    }
    pub fn insert(&mut self, key: String, value: RegressionReport) {
        if self.cache.len() >= self.max_size {
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, value);
    }
    pub fn get_statistics(&self) -> DetectionStatistics {
        DetectionStatistics {
            total_detections: self.hit_count + self.miss_count,
            true_positives: self.hit_count,
            false_positives: 0,
            false_negatives: 0,
            precision: if self.hit_count + self.miss_count > 0 {
                self.hit_count as f64 / (self.hit_count + self.miss_count) as f64
            } else {
                0.0
            },
            recall: 1.0,
            f1_score: 1.0,
            detection_latency: Duration::from_millis(100),
        }
    }
}
/// Trend analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisReport {
    pub trend_id: String,
    pub analysis_period: Duration,
    pub trend_summary: TrendSummary,
    pub metric_trends: HashMap<String, MetricTrend>,
    pub predictions: Vec<TrendPrediction>,
}
/// Maintenance windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceWindow {
    pub window_name: String,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub recurring: bool,
    pub affected_systems: Vec<String>,
}
/// Configuration for regression detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetectorConfig {
    pub sensitivity: DetectionSensitivity,
    pub regression_thresholds: RegressionThresholds,
    pub anomaly_threshold: f64,
    pub monitoring_period: Duration,
    pub enable_caching: bool,
    pub cache_size: usize,
    pub continuous_monitoring: bool,
}
/// Types of regressions
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum RegressionType {
    PerformanceDegradation,
    AccuracyDrop,
    MemoryIncrease,
    LatencyIncrease,
    ThroughputDecrease,
    QualityDegradation,
    TrendDegradation,
    AnomalyRegression,
    Custom(String),
}
/// Regression recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionRecommendation {
    pub recommendation_id: String,
    pub recommendation_type: RegressionRecommendationType,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub affected_benchmarks: Vec<String>,
    pub root_cause_analysis: Vec<String>,
    pub remediation_steps: Vec<String>,
    pub expected_timeline: Duration,
    pub confidence: f64,
    pub estimated_effort: ImplementationEffort,
    pub expected_impact: RegressionImpact,
}
/// Baseline comparisons for regression detection
#[derive(Debug)]
pub struct BaselineComparisons {
    pub(super) comparison_methods: Vec<BaselineComparisonMethod>,
    pub(super) baseline_manager: BaselineManager,
    pub(super) significance_testing: SignificanceTesting,
    pub(super) effect_size_analysis: EffectSizeAnalysis,
}
impl BaselineComparisons {
    pub fn new() -> Self {
        Self {
            comparison_methods: vec![BaselineComparisonMethod::DirectComparison],
            baseline_manager: BaselineManager::new(),
            significance_testing: SignificanceTesting::new(),
            effect_size_analysis: EffectSizeAnalysis::new(),
        }
    }
}
/// Suppression rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionRule {
    pub rule_name: String,
    pub condition: SuppressionCondition,
    pub duration: Option<Duration>,
    pub enabled: bool,
}
/// Pattern recognition for alert suppression
#[derive(Debug)]
pub struct PatternRecognition {
    pub(super) alert_clustering: bool,
    pub(super) temporal_patterns: bool,
    pub(super) correlation_analysis: bool,
}
impl PatternRecognition {
    pub fn new() -> Self {
        Self {
            alert_clustering: false,
            temporal_patterns: false,
            correlation_analysis: false,
        }
    }
}
/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub disk_utilization: f64,
    pub network_utilization: f64,
    pub load_average: f64,
}
/// Escalation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationStep {
    pub level: u32,
    pub delay: Duration,
    pub notification_channels: Vec<String>,
    pub acknowledgment_required: bool,
}
/// Remediation actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationAction {
    pub action_id: String,
    pub action_type: RemediationActionType,
    pub description: String,
    pub priority: RemediationPriority,
    pub estimated_impact: f64,
    pub implementation_cost: ImplementationCost,
    pub risk_level: RiskLevel,
}
/// Threshold management for regressions
#[derive(Debug)]
pub struct ThresholdManagement {
    pub(super) threshold_types: Vec<ThresholdType>,
    pub(super) adaptive_thresholds: AdaptiveThresholds,
    pub(super) threshold_history: Vec<ThresholdHistoryEntry>,
}
impl ThresholdManagement {
    pub fn new() -> Self {
        Self {
            threshold_types: vec![ThresholdType::Static, ThresholdType::Adaptive],
            adaptive_thresholds: AdaptiveThresholds::new(),
            threshold_history: Vec::new(),
        }
    }
    pub fn update_sensitivity(&mut self, sensitivity: &DetectionSensitivity) {
        self.adaptive_thresholds.update_sensitivity(sensitivity);
    }
    pub fn update_thresholds(&mut self, thresholds: RegressionThresholds) {
        let entry = ThresholdHistoryEntry {
            timestamp: SystemTime::now(),
            thresholds,
            change_reason: "Manual update".to_string(),
        };
        self.threshold_history.push(entry);
    }
}
/// Notification channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Email,
    Slack,
    Teams,
    Webhook,
    SMS,
    Custom(String),
}
/// Recent changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentChange {
    pub change_type: ChangeType,
    pub description: String,
    pub timestamp: SystemTime,
    pub correlation_score: f64,
}
/// Regression alert system
#[derive(Debug)]
pub struct RegressionAlertSystem {
    pub(super) alert_rules: Vec<RegressionAlertRule>,
    pub(super) notification_channels: Vec<NotificationChannel>,
    pub(super) escalation_policies: Vec<EscalationPolicy>,
    pub(super) alert_suppression: AlertSuppression,
}
impl RegressionAlertSystem {
    pub fn new() -> Self {
        Self {
            alert_rules: vec![
                RegressionAlertRule {
                    rule_name: "Performance degradation".to_string(),
                    condition: RegressionCondition::PerformanceDegradation(0.1),
                    severity: AlertSeverity::Warning,
                    threshold: 0.1,
                    consecutive_failures: 2,
                    enabled: true,
                },
                RegressionAlertRule {
                    rule_name: "Critical performance drop".to_string(),
                    condition: RegressionCondition::PerformanceDegradation(0.25),
                    severity: AlertSeverity::Critical,
                    threshold: 0.25,
                    consecutive_failures: 1,
                    enabled: true,
                },
            ],
            notification_channels: vec![NotificationChannel {
                channel_name: "email_alerts".to_string(),
                channel_type: NotificationChannelType::Email,
                configuration: HashMap::new(),
                enabled: true,
                rate_limiting: RateLimiting {
                    max_alerts_per_hour: 10,
                    burst_allowance: 3,
                    cooldown_period: Duration::from_secs(300),
                },
            }],
            escalation_policies: Vec::new(),
            alert_suppression: AlertSuppression::new(),
        }
    }
}
/// Adaptive thresholds
#[derive(Debug)]
pub struct AdaptiveThresholds {
    pub(super) learning_rate: f64,
    pub(super) window_size: usize,
    pub(super) min_samples: usize,
    pub(super) confidence_level: f64,
}
impl AdaptiveThresholds {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.1,
            window_size: 100,
            min_samples: 10,
            confidence_level: 0.95,
        }
    }
    pub fn update_sensitivity(&mut self, sensitivity: &DetectionSensitivity) {
        match sensitivity {
            DetectionSensitivity::Low => self.confidence_level = 0.99,
            DetectionSensitivity::Medium => self.confidence_level = 0.95,
            DetectionSensitivity::High => self.confidence_level = 0.90,
            DetectionSensitivity::Custom(level) => self.confidence_level = *level,
        }
    }
}
/// Regression impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionImpact {
    pub performance_improvement: f64,
    pub risk_reduction: f64,
    pub cost_impact: CostImpact,
    pub timeline_impact: Duration,
}
/// Early warning indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyWarning {
    pub warning_id: String,
    pub warning_type: WarningType,
    pub description: String,
    pub severity: WarningSeverity,
    pub confidence: f64,
    pub estimated_time_to_impact: Duration,
    pub recommended_actions: Vec<String>,
}
/// Regression detection errors
#[derive(Debug, thiserror::Error)]
pub enum RegressionError {
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    #[error("Threshold error: {0}")]
    ThresholdError(String),
    #[error("Alert error: {0}")]
    AlertError(String),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Statistical error: {0}")]
    StatisticalError(String),
}
#[derive(Debug, Clone)]
pub struct CorrelationAnalysisResult {
    pub potential_causes: Vec<PotentialCause>,
}
/// Continuous monitoring report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousMonitoringReport {
    pub monitoring_id: String,
    pub timestamp: SystemTime,
    pub regression_report: RegressionReport,
    pub trend_analysis: TrendAnalysisReport,
    pub health_score: SystemHealthScore,
    pub early_warnings: Vec<EarlyWarning>,
    pub monitoring_period: Duration,
}
/// Remediation action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationActionType {
    ImmediateFix,
    ShortTermWorkaround,
    LongTermSolution,
    PreventiveMeasure,
    MonitoringImprovement,
}
/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    PerformanceRegression,
    QualityDegradation,
    ResourceExhaustion,
    SystemFailure,
    TrendDeviation,
    AnomalyDetected,
    Custom(String),
}
