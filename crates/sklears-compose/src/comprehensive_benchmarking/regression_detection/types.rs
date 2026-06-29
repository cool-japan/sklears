//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::config_types::*;
use super::super::performance_analysis::ImpactLevel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::types_20::{
    AlertType, EscalationStep, MaintenanceWindow, NotificationChannelType, PatternRecognition,
    RecentChange, RegressionRecommendation, RegressionType, RemediationAction, SuppressionRule,
    SystemMetrics, UserImpactAssessment,
};

/// Suppression conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuppressionCondition {
    MaintenanceWindow,
    KnownIssue,
    DeploymentWindow,
    HighVolumeAlerts,
    Custom(String),
}
/// Smart suppression for intelligent alert filtering
#[derive(Debug)]
pub struct SmartSuppression {
    pub(super) machine_learning_enabled: bool,
    pub(super) pattern_recognition: PatternRecognition,
    pub(super) correlation_threshold: f64,
}
impl SmartSuppression {
    pub fn new() -> Self {
        Self {
            machine_learning_enabled: false,
            pattern_recognition: PatternRecognition::new(),
            correlation_threshold: 0.8,
        }
    }
}
/// Potential causes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialCause {
    pub cause_id: String,
    pub cause_type: CauseType,
    pub description: String,
    pub likelihood: f64,
    pub evidence: Vec<Evidence>,
    pub investigation_required: bool,
}
/// Evidence types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    Statistical,
    Temporal,
    Correlational,
    Environmental,
    Observational,
    Historical,
}
/// Investigation priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvestigationPriority {
    Immediate,
    High,
    Medium,
    Low,
}
/// Regression alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    pub alert_id: String,
    pub regression_id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub triggered_at: SystemTime,
    pub resolved: bool,
    pub escalation_level: u32,
    pub notification_channels: Vec<String>,
    pub acknowledgment_required: bool,
    pub auto_resolution_timeout: Option<Duration>,
}
/// Regression context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionContext {
    pub environmental_factors: Vec<EnvironmentalFactor>,
    pub recent_changes: Vec<RecentChange>,
    pub system_metrics: SystemMetrics,
    pub additional_info: HashMap<String, String>,
}
/// Threshold types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdType {
    Static,
    Adaptive,
    PercentileBased,
    MachineLearning,
    Custom(String),
}
/// Detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionStatistics {
    pub total_detections: u64,
    pub true_positives: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub detection_latency: Duration,
}
/// Detection sensitivity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionSensitivity {
    Low,
    Medium,
    High,
    Custom(f64),
}
/// Baseline comparison methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineComparisonMethod {
    DirectComparison,
    StatisticalTest,
    DistributionComparison,
    TrendComparison,
    Custom(String),
}
/// Significance testing for regressions
#[derive(Debug)]
pub struct SignificanceTesting {
    pub(super) test_methods: Vec<SignificanceTestMethod>,
    pub(super) significance_level: f64,
    pub(super) multiple_testing_correction: bool,
}
impl SignificanceTesting {
    pub fn new() -> Self {
        Self {
            test_methods: vec![SignificanceTestMethod::TTest],
            significance_level: 0.05,
            multiple_testing_correction: true,
        }
    }
}
/// Metric trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub expected_value: f64,
    pub degradation_rate: f64,
    pub confidence: f64,
    pub trend_start_time: SystemTime,
    pub consecutive_periods: u32,
}
/// Regression detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionDetectionAlgorithm {
    StatisticalRegression,
    ChangePointDetection,
    TrendAnalysis,
    AnomalyDetection,
    MachineLearning,
    Custom(String),
}
/// Trend prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPrediction {
    pub metric_name: String,
    pub predicted_values: Vec<f64>,
    pub prediction_horizon: Duration,
    pub confidence_interval: (f64, f64),
    pub prediction_accuracy: f64,
}
/// Regression alert rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlertRule {
    pub rule_name: String,
    pub condition: RegressionCondition,
    pub severity: AlertSeverity,
    pub threshold: f64,
    pub consecutive_failures: u32,
    pub enabled: bool,
}
/// Escalation policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    pub policy_name: String,
    pub escalation_steps: Vec<EscalationStep>,
    pub max_escalation_level: u32,
}
/// Warning severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningSeverity {
    Advisory,
    Watch,
    Warning,
    Critical,
}
/// System health score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthScore {
    pub overall_score: f64,
    pub component_scores: HashMap<String, f64>,
    pub health_trend: TrendDirection,
    pub critical_issues: usize,
    pub warnings: usize,
}
/// Significance test methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignificanceTestMethod {
    TTest,
    MannWhitneyU,
    KolmogorovSmirnov,
    Custom(String),
}
/// Effect size measures
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum EffectSizeMeasure {
    CohensD,
    HedgesG,
    GlasssDelta,
    PercentageChange,
    Custom(String),
}
/// Types of regression recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionRecommendationType {
    OptimizationNeeded,
    InvestigateRootCause,
    RollbackChanges,
    ScaleResources,
    UpdateBaseline,
    EnvironmentStabilization,
    Custom(String),
}
/// Overall regression severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverallRegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}
/// Rate limiting for notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiting {
    pub max_alerts_per_hour: u32,
    pub burst_allowance: u32,
    pub cooldown_period: Duration,
}
/// Detected regression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedRegression {
    pub regression_id: String,
    pub benchmark_id: String,
    pub metric_name: String,
    pub regression_type: RegressionType,
    pub severity: RegressionSeverity,
    pub current_value: f64,
    pub expected_value: f64,
    pub degradation_percentage: f64,
    pub detection_confidence: f64,
    pub first_detected: SystemTime,
    pub consecutive_failures: u32,
    pub detection_method: String,
    pub context: RegressionContext,
}
/// Cause types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CauseType {
    CodeRegression,
    ConfigurationChange,
    EnvironmentalChange,
    ResourceConstraint,
    DataQualityIssue,
    InfrastructureProblem,
    ExternalDependency,
    Unknown,
}
/// Regression metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionMetadata {
    pub detector_version: String,
    pub detection_parameters: HashMap<String, String>,
    pub analysis_duration: Duration,
    pub data_quality_score: f64,
}
/// Change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    CodeChange,
    ConfigurationChange,
    InfrastructureChange,
    DataChange,
    EnvironmentChange,
    Custom(String),
}
/// Regression severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}
/// Warning types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningType {
    PerformanceTrend,
    ResourceUtilization,
    QualityDegradation,
    SystemStability,
    Custom(String),
}
/// Severity assessment for regressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverityAssessment {
    pub overall_severity: OverallRegressionSeverity,
    pub critical_regressions: usize,
    pub high_severity_regressions: usize,
    pub medium_severity_regressions: usize,
    pub low_severity_regressions: usize,
    pub total_regressions: usize,
    pub risk_score: f64,
    pub business_impact: BusinessImpactAssessment,
    pub user_impact: UserImpactAssessment,
    pub estimated_recovery_time: Duration,
}
/// Trend summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSummary {
    pub overall_trend: TrendDirection,
    pub trend_strength: f64,
    pub confidence: f64,
    pub significant_changes: usize,
}
/// Alert suppression settings
#[derive(Debug)]
pub struct AlertSuppression {
    pub(super) suppression_rules: Vec<SuppressionRule>,
    pub(super) maintenance_windows: Vec<MaintenanceWindow>,
    pub(super) smart_suppression: SmartSuppression,
}
impl AlertSuppression {
    pub fn new() -> Self {
        Self {
            suppression_rules: Vec::new(),
            maintenance_windows: Vec::new(),
            smart_suppression: SmartSuppression::new(),
        }
    }
}
/// Root cause analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCauseAnalysis {
    pub analysis_id: String,
    pub analysis_timestamp: SystemTime,
    pub potential_causes: Vec<PotentialCause>,
    pub confidence_scores: HashMap<String, f64>,
    pub investigation_steps: Vec<InvestigationStep>,
    pub remediation_priority: Vec<RemediationAction>,
}
/// Regression conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionCondition {
    PerformanceDegradation(f64),
    AccuracyDrop(f64),
    LatencyIncrease(f64),
    MemoryIncrease(f64),
    ThroughputDecrease(f64),
    Custom(String),
}
/// Investigation steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestigationStep {
    pub step_id: String,
    pub description: String,
    pub priority: InvestigationPriority,
    pub estimated_time: Duration,
    pub required_skills: Vec<String>,
    pub tools_needed: Vec<String>,
}
/// Remediation priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationPriority {
    Critical,
    High,
    Medium,
    Low,
}
/// Implementation cost assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationCost {
    pub time_cost: Duration,
    pub resource_cost: f64,
    pub complexity_score: f64,
    pub risk_factor: f64,
}
/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub channel_name: String,
    pub channel_type: NotificationChannelType,
    pub configuration: HashMap<String, String>,
    pub enabled: bool,
    pub rate_limiting: RateLimiting,
}
/// Evidence for potential causes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub evidence_type: EvidenceType,
    pub description: String,
    pub strength: f64,
    pub source: String,
}
/// Comprehensive regression report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    pub report_id: String,
    pub detection_timestamp: SystemTime,
    pub total_benchmarks_analyzed: usize,
    pub regressions_detected: usize,
    pub detected_regressions: Vec<DetectedRegression>,
    pub severity_assessment: SeverityAssessment,
    pub alerts_triggered: Vec<RegressionAlert>,
    pub recommendations: Vec<RegressionRecommendation>,
    pub root_cause_analysis: RootCauseAnalysis,
    pub confidence_score: f64,
    pub metadata: RegressionMetadata,
}
/// Threshold history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdHistoryEntry {
    pub timestamp: SystemTime,
    pub thresholds: RegressionThresholds,
    pub change_reason: String,
}
/// Business impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpactAssessment {
    pub revenue_impact: f64,
    pub customer_impact: f64,
    pub operational_impact: f64,
    pub reputation_impact: f64,
}
/// Effect size analysis for regressions
#[derive(Debug)]
pub struct EffectSizeAnalysis {
    pub(super) effect_size_measures: Vec<EffectSizeMeasure>,
    pub(super) practical_significance_thresholds: HashMap<EffectSizeMeasure, f64>,
}
impl EffectSizeAnalysis {
    pub fn new() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(EffectSizeMeasure::CohensD, 0.5);
        thresholds.insert(EffectSizeMeasure::HedgesG, 0.5);
        Self {
            effect_size_measures: vec![EffectSizeMeasure::CohensD, EffectSizeMeasure::HedgesG],
            practical_significance_thresholds: thresholds,
        }
    }
}
/// Environmental factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalFactor {
    pub factor_type: String,
    pub description: String,
    pub impact_level: ImpactLevel,
    pub confidence: f64,
}
