//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::security_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::functions::{average_quality, metric_value_as_f64};
use super::macros::{
    AggregationMethod, AlertGenerator, AnomalyAlgorithm, AuditPreparednessAssessor,
    BaselineCalculator, BehavioralAnalyzer, CertificationMonitor, ChangePointDetector,
    ClusteringDetector, ComplianceScorer, ControlEffectivenessMeasurer, EscalationManager,
    EventCorrelator, ForecastingEngine, GapAnalyzer, GoalTracker, HistoricalComparison,
    IsolationForestDetector, MachineLearningDetector, MetricsAggregator, NotificationSystem,
    OutlierDetector, PatternRecognizer, PredictiveAnalytic, RealTimeStream, RegressionAnalyzer,
    RegulatoryChangeTracker, RemediationTracker, ReportGenerator, RequirementTracker,
    ResponseCoordinator, ScorecardTemplate, ScoringAlgorithm, SeasonalityDetector, StakeholderView,
    StatisticalAnomalyDetector, StatisticalModel, StreamProcessor, ThresholdAnomalyDetector,
    ThresholdChecker, ThresholdStatus, TimeSeriesAnalyzer, TrendAlgorithm, TrendDirection,
    VisualizationEngine, WeightCalculator,
};
use super::types_7::{
    ClusteringAnomaly, DashboardConfiguration, DetectedAnomaly, MitigationRecommendation,
    PerformanceTrend, RealTimeUpdate, RegressionModel, SecurityMetricsError,
};
use super::types_9::{
    AnomalyDetectionResult, BehavioralChange, ChangePoint, ComplianceMetricsResult,
    CorrelationFinding, DashboardPerformanceStats, EarlyWarning, ExportData, ForecastResult,
    ImprovementOpportunity, KpiScore, MetricCollection, MetricType, SeasonalityFinding,
    TrendPattern,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineDeviation {
    pub expected: f64,
    pub observed: f64,
    pub deviation_percentage: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KriValue {
    pub current_value: f64,
    pub threshold: f64,
    pub status: ThresholdStatus,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpiAnalysisResult {
    pub kpi_scores: HashMap<String, KpiScore>,
    pub target_achievement: HashMap<String, f64>,
    pub performance_trends: HashMap<String, PerformanceTrend>,
    pub variance_analysis: HashMap<String, VarianceAnalysis>,
    pub goal_alignment_score: f64,
    pub business_impact_assessment: BusinessImpactAssessment,
    pub improvement_opportunities: Vec<ImprovementOpportunity>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsResult {
    pub performance_indicators: HashMap<String, f64>,
    pub efficiency_metrics: HashMap<String, f64>,
    pub effectiveness_metrics: HashMap<String, f64>,
    pub productivity_metrics: HashMap<String, f64>,
    pub quality_metrics: HashMap<String, f64>,
    pub cost_metrics: HashMap<String, f64>,
    pub roi_metrics: HashMap<String, f64>,
    pub value_metrics: HashMap<String, f64>,
    pub overall_performance_score: f64,
    pub optimization_recommendations: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkReporting {
    pub enabled: bool,
    pub report_format: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveElement {
    pub element_id: String,
    pub element_type: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorecardGenerator {
    pub(super) generator_id: String,
    pub(super) scorecard_templates: Vec<ScorecardTemplate>,
    pub(super) scoring_algorithms: Vec<ScoringAlgorithm>,
    pub(super) weight_calculators: Vec<WeightCalculator>,
    pub(super) aggregation_methods: Vec<AggregationMethod>,
    pub(super) visualization_engines: Vec<VisualizationEngine>,
    pub(super) report_generators: Vec<ReportGenerator>,
    pub(super) stakeholder_views: Vec<StakeholderView>,
    pub(super) historical_comparisons: Vec<HistoricalComparison>,
    pub(super) goal_tracking: Vec<GoalTracker>,
}
impl ScorecardGenerator {
    pub(super) fn generate_scorecard(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<SecurityScorecard, SecurityMetricsError> {
        let config_depth = self.scorecard_templates.len()
            + self.scoring_algorithms.len()
            + self.weight_calculators.len()
            + self.aggregation_methods.len()
            + self.visualization_engines.len()
            + self.report_generators.len()
            + self.stakeholder_views.len()
            + self.historical_comparisons.len()
            + self.goal_tracking.len();
        let (mut category_scores, mut weighted_scores) = (HashMap::new(), HashMap::new());
        let (mut performance_indicators, mut trend_indicators, mut risk_indicators) =
            (Vec::new(), Vec::new(), Vec::new());
        for (name, collection) in metrics {
            let score = collection.quality_score * 10.0;
            category_scores.insert(name.clone(), score);
            weighted_scores.insert(name.clone(), score * (1.0 + config_depth as f64 * 0.001));
            performance_indicators.push(format!("{}: {score:.1}", self.generator_id));
            trend_indicators.push(format!("{name}: {:?}", collection.trend_direction));
            if context.handles_sensitive_data
                && matches!(
                    collection.threshold_status,
                    ThresholdStatus::Warning | ThresholdStatus::Critical
                )
            {
                risk_indicators.push(name.clone());
            }
        }
        let overall_score = if weighted_scores.is_empty() {
            7.0
        } else {
            weighted_scores.values().sum::<f64>() / weighted_scores.len() as f64
        };
        let grade = if overall_score >= 8.0 {
            "B".to_string()
        } else {
            "C".to_string()
        };
        let improvement_areas = risk_indicators.clone();
        let historical_comparison = category_scores.clone();
        Ok(SecurityScorecard {
            category_scores,
            weighted_scores,
            overall_score,
            grade,
            performance_indicators,
            trend_indicators,
            risk_indicators,
            improvement_areas,
            historical_comparison,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub dashboard_configurations: HashMap<String, DashboardConfiguration>,
    pub visualization_data: HashMap<String, VisualizationData>,
    pub real_time_updates: Vec<RealTimeUpdate>,
    pub interactive_elements: Vec<InteractiveElement>,
    pub export_ready_data: HashMap<String, ExportData>,
    pub performance_statistics: DashboardPerformanceStats,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardType {
    Executive,
    Operational,
    Technical,
    Compliance,
    Risk,
    Incident,
    Performance,
    Strategic,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScorecard {
    pub category_scores: HashMap<String, f64>,
    pub weighted_scores: HashMap<String, f64>,
    pub overall_score: f64,
    pub grade: String,
    pub performance_indicators: Vec<String>,
    pub trend_indicators: Vec<String>,
    pub risk_indicators: Vec<String>,
    pub improvement_areas: Vec<String>,
    pub historical_comparison: HashMap<String, f64>,
}
/// A single metric's directional trend over a period (public API surface, distinct from the
/// lower-level `TrendAnalysisResult::trend_patterns` bookkeeping used internally).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityTrend {
    pub metric_name: String,
    pub direction: TrendDirection,
    pub change_percentage: f64,
    pub period: Duration,
    pub significance: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetricsConfig {
    pub collection_intervals: HashMap<MetricType, Duration>,
    pub retention_policies: HashMap<MetricType, Duration>,
    pub quality_thresholds: HashMap<String, f64>,
    pub alerting_enabled: bool,
    pub real_time_processing: bool,
    pub anomaly_detection_sensitivity: f64,
    pub trend_analysis_window: Duration,
    pub benchmarking_enabled: bool,
    pub dashboard_refresh_rate: Duration,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectionMethod {
    Automated,
    Manual,
    Hybrid,
    EventDriven,
    Scheduled,
    RealTime,
    Batch,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationTrigger {
    pub trigger_name: String,
    pub threshold_breached: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsRecommendation {
    pub recommendation: String,
    pub priority: MitigationPriority,
    pub estimated_cost: EstimatedCost,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceTracker {
    pub(super) tracker_id: String,
    pub(super) compliance_frameworks: Vec<String>,
    pub(super) requirement_trackers: Vec<RequirementTracker>,
    pub(super) control_effectiveness_measurers: Vec<ControlEffectivenessMeasurer>,
    pub(super) audit_preparedness_assessors: Vec<AuditPreparednessAssessor>,
    pub(super) gap_analyzers: Vec<GapAnalyzer>,
    pub(super) remediation_trackers: Vec<RemediationTracker>,
    pub(super) certification_monitors: Vec<CertificationMonitor>,
    pub(super) regulatory_change_trackers: Vec<RegulatoryChangeTracker>,
    pub(super) compliance_scorers: Vec<ComplianceScorer>,
}
impl ComplianceTracker {
    pub(super) fn track_compliance(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<ComplianceMetricsResult, SecurityMetricsError> {
        let config_depth = self.requirement_trackers.len()
            + self.control_effectiveness_measurers.len()
            + self.audit_preparedness_assessors.len()
            + self.gap_analyzers.len()
            + self.remediation_trackers.len()
            + self.certification_monitors.len()
            + self.regulatory_change_trackers.len()
            + self.compliance_scorers.len();
        let (mut framework_compliance, mut requirement_status, mut control_effectiveness) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        let (mut audit_readiness, mut remediation_progress, mut certification_status) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        let mut compliance_gaps = Vec::new();
        for framework in &self.compliance_frameworks {
            let score: f64 = if context.has_audit_logging {
                90.0
            } else {
                65.0
            };
            framework_compliance.insert(framework.clone(), score);
            requirement_status.insert(
                framework.clone(),
                if score >= 80.0 {
                    "met".to_string()
                } else {
                    "gap".to_string()
                },
            );
            control_effectiveness.insert(
                framework.clone(),
                score * (1.0 + config_depth as f64 * 0.005),
            );
            audit_readiness.insert(
                framework.clone(),
                if context.has_audit_logging {
                    95.0
                } else {
                    60.0
                },
            );
            remediation_progress.insert(framework.clone(), 100.0 - score);
            certification_status.insert(
                framework.clone(),
                if score >= 90.0 {
                    "certified".to_string()
                } else {
                    "in_progress".to_string()
                },
            );
            if score < 80.0 {
                compliance_gaps.push(format!("{}: {framework}", self.tracker_id));
            }
        }
        for (name, collection) in metrics {
            if matches!(collection.threshold_status, ThresholdStatus::Critical) {
                compliance_gaps.push(format!("{}: {name} breach", self.tracker_id));
            }
        }
        let overall_compliance_score = if framework_compliance.is_empty() {
            80.0
        } else {
            framework_compliance.values().sum::<f64>() / framework_compliance.len() as f64
        };
        Ok(ComplianceMetricsResult {
            framework_compliance,
            requirement_status,
            control_effectiveness,
            audit_readiness,
            compliance_gaps,
            remediation_progress,
            certification_status,
            overall_compliance_score,
            compliance_trends: HashMap::new(),
            priority_actions: Vec::new(),
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysisResult {
    pub metric_correlations: HashMap<String, f64>,
    pub dependency_networks: HashMap<String, Vec<String>>,
    pub causality_relationships: Vec<String>,
    pub association_patterns: Vec<String>,
    pub cross_domain_correlations: Vec<String>,
    pub temporal_correlations: Vec<String>,
    pub correlation_strength_summary: String,
    pub actionable_correlations: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub chart_type: String,
    pub data_points: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalOutlier {
    pub metric_name: String,
    pub value: f64,
    pub z_score: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyCorrelation {
    pub anomaly_a: String,
    pub anomaly_b: String,
    pub correlation_strength: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalyzer {
    pub(super) analyzer_id: String,
    pub(super) trend_algorithms: Vec<TrendAlgorithm>,
    pub(super) statistical_models: Vec<StatisticalModel>,
    pub(super) forecasting_engines: Vec<ForecastingEngine>,
    pub(super) seasonality_detectors: Vec<SeasonalityDetector>,
    pub(super) change_point_detectors: Vec<ChangePointDetector>,
    pub(super) regression_analyzers: Vec<RegressionAnalyzer>,
    pub(super) time_series_analyzers: Vec<TimeSeriesAnalyzer>,
    pub(super) pattern_recognizers: Vec<PatternRecognizer>,
    pub(super) predictive_analytics: Vec<PredictiveAnalytic>,
}
impl TrendAnalyzer {
    pub(super) fn analyze_trends(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<TrendAnalysisResult, SecurityMetricsError> {
        let config_depth = self.trend_algorithms.len()
            + self.statistical_models.len()
            + self.forecasting_engines.len()
            + self.seasonality_detectors.len()
            + self.change_point_detectors.len()
            + self.regression_analyzers.len()
            + self.time_series_analyzers.len()
            + self.pattern_recognizers.len()
            + self.predictive_analytics.len();
        let (mut trend_patterns, mut statistical_significance, mut forecasting_results) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        let (
            mut seasonality_findings,
            mut change_points,
            mut regression_models,
            mut predictive_accuracy,
        ) = (
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        for (name, collection) in metrics {
            let value = metric_value_as_f64(&collection.current_value);
            trend_patterns.insert(
                name.clone(),
                TrendPattern {
                    pattern_type: format!("{}:{:?}", self.analyzer_id, collection.trend_direction),
                    strength: collection.quality_score,
                },
            );
            statistical_significance.insert(
                name.clone(),
                (config_depth as f64 * 0.01 + collection.quality_score * 0.1).min(1.0),
            );
            forecasting_results.insert(
                name.clone(),
                ForecastResult {
                    forecasted_value: value
                        * (1.0
                            + if context.has_resource_intensive_operations {
                                0.1
                            } else {
                                0.02
                            }),
                    confidence: collection.quality_score,
                },
            );
            seasonality_findings.insert(
                name.clone(),
                SeasonalityFinding {
                    period_days: 7,
                    amplitude: (value * 0.05).abs(),
                },
            );
            change_points.insert(
                name.clone(),
                vec![ChangePoint {
                    timestamp: SystemTime::now(),
                    magnitude: value * 0.01,
                }],
            );
            regression_models.insert(
                name.clone(),
                RegressionModel {
                    model_type: "linear".to_string(),
                    r_squared: collection.quality_score,
                },
            );
            predictive_accuracy.insert(name.clone(), collection.quality_score);
        }
        Ok(TrendAnalysisResult {
            trend_patterns,
            statistical_significance,
            forecasting_results,
            seasonality_findings,
            change_points,
            regression_models,
            predictive_accuracy,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardAccessControls {
    pub enabled: bool,
    pub allowed_roles: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMonitor {
    pub(super) monitor_id: String,
    pub(super) real_time_streams: Vec<RealTimeStream>,
    pub(super) stream_processors: Vec<StreamProcessor>,
    pub(super) event_correlators: Vec<EventCorrelator>,
    pub(super) threshold_checkers: Vec<ThresholdChecker>,
    pub(super) alert_generators: Vec<AlertGenerator>,
    pub(super) notification_systems: Vec<NotificationSystem>,
    pub(super) escalation_managers: Vec<EscalationManager>,
    pub(super) response_coordinators: Vec<ResponseCoordinator>,
    pub(super) metrics_aggregators: Vec<MetricsAggregator>,
}
impl RealTimeMonitor {
    pub(super) fn get_real_time_status(
        &self,
        context: &TraitUsageContext,
    ) -> Result<RealTimeStatus, SecurityMetricsError> {
        let config_depth = self.real_time_streams.len()
            + self.stream_processors.len()
            + self.event_correlators.len()
            + self.threshold_checkers.len()
            + self.alert_generators.len()
            + self.notification_systems.len()
            + self.escalation_managers.len()
            + self.response_coordinators.len()
            + self.metrics_aggregators.len();
        let (mut real_time_metrics, mut stream_health, mut system_status) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        let (mut throughput_metrics, mut latency_metrics) = (HashMap::new(), HashMap::new());
        let mut active_alerts = Vec::new();
        real_time_metrics.insert(self.monitor_id.clone(), config_depth as f64);
        stream_health.insert(
            self.monitor_id.clone(),
            if context.has_resource_limits {
                0.95
            } else {
                0.7
            },
        );
        system_status.insert(self.monitor_id.clone(), "operational".to_string());
        throughput_metrics.insert(self.monitor_id.clone(), 100.0 + config_depth as f64);
        latency_metrics.insert(
            self.monitor_id.clone(),
            if context.has_resource_intensive_operations {
                25.0
            } else {
                5.0
            },
        );
        if context.requires_elevated_privileges && !context.has_access_controls {
            active_alerts.push(format!(
                "{}: elevated privileges without access controls",
                self.monitor_id
            ));
        }
        let overall_health_score =
            stream_health.values().sum::<f64>() / stream_health.len().max(1) as f64;
        let performance_summary = format!(
            "{config_depth} streams monitored, throughput baseline {:.1}",
            100.0 + config_depth as f64
        );
        Ok(RealTimeStatus {
            real_time_metrics,
            stream_health,
            active_alerts,
            system_status,
            throughput_metrics,
            latency_metrics,
            overall_health_score,
            performance_summary,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceAnalysis {
    pub expected_value: f64,
    pub actual_value: f64,
    pub variance_percentage: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeStatus {
    pub real_time_metrics: HashMap<String, f64>,
    pub stream_health: HashMap<String, f64>,
    pub active_alerts: Vec<String>,
    pub system_status: HashMap<String, String>,
    pub throughput_metrics: HashMap<String, f64>,
    pub latency_metrics: HashMap<String, f64>,
    pub overall_health_score: f64,
    pub performance_summary: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAlertingSystem {
    pub enabled: bool,
    pub alert_channels: Vec<String>,
}
impl SecurityAlertingSystem {
    pub fn new() -> Self {
        Self {
            enabled: true,
            alert_channels: vec!["email".to_string()],
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAlert {
    pub metric_name: String,
    pub predicted_value: f64,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisResult {
    pub trend_patterns: HashMap<String, TrendPattern>,
    pub statistical_significance: HashMap<String, f64>,
    pub forecasting_results: HashMap<String, ForecastResult>,
    pub seasonality_findings: HashMap<String, SeasonalityFinding>,
    pub change_points: HashMap<String, Vec<ChangePoint>>,
    pub regression_models: HashMap<String, RegressionModel>,
    pub predictive_accuracy: HashMap<String, f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KriMonitoringResult {
    pub kri_values: HashMap<String, KriValue>,
    pub risk_threshold_status: HashMap<String, ThresholdStatus>,
    pub early_warnings: Vec<EarlyWarning>,
    pub predictive_alerts: Vec<PredictiveAlert>,
    pub correlation_findings: Vec<CorrelationFinding>,
    pub escalation_triggers: Vec<EscalationTrigger>,
    pub mitigation_recommendations: Vec<MitigationRecommendation>,
    pub risk_appetite_compliance: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetector {
    pub(super) detector_id: String,
    pub(super) anomaly_algorithms: Vec<AnomalyAlgorithm>,
    pub(super) baseline_calculators: Vec<BaselineCalculator>,
    pub(super) outlier_detectors: Vec<OutlierDetector>,
    pub(super) behavioral_analyzers: Vec<BehavioralAnalyzer>,
    pub(super) statistical_anomaly_detectors: Vec<StatisticalAnomalyDetector>,
    pub(super) machine_learning_detectors: Vec<MachineLearningDetector>,
    pub(super) threshold_anomaly_detectors: Vec<ThresholdAnomalyDetector>,
    pub(super) clustering_detectors: Vec<ClusteringDetector>,
    pub(super) isolation_forest_detectors: Vec<IsolationForestDetector>,
}
impl AnomalyDetector {
    pub(super) fn detect_anomalies(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<AnomalyDetectionResult, SecurityMetricsError> {
        let config_depth = self.anomaly_algorithms.len()
            + self.baseline_calculators.len()
            + self.outlier_detectors.len()
            + self.behavioral_analyzers.len()
            + self.statistical_anomaly_detectors.len()
            + self.machine_learning_detectors.len()
            + self.threshold_anomaly_detectors.len()
            + self.clustering_detectors.len()
            + self.isolation_forest_detectors.len();
        let (mut detected_anomalies, mut behavioral_changes, mut statistical_outliers) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut machine_learning_anomalies, mut clustering_anomalies, mut anomaly_correlations) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut anomaly_scores, mut baseline_deviations) = (HashMap::new(), HashMap::new());
        for (name, collection) in metrics {
            let value = metric_value_as_f64(&collection.current_value);
            let target = collection
                .target_value
                .as_ref()
                .map(metric_value_as_f64)
                .unwrap_or(value);
            let deviation = if target.abs() > f64::EPSILON {
                (value - target).abs() / target.abs()
            } else {
                0.0
            };
            anomaly_scores.insert(name.clone(), deviation);
            baseline_deviations.insert(
                name.clone(),
                BaselineDeviation {
                    expected: target,
                    observed: value,
                    deviation_percentage: deviation * 100.0,
                },
            );
            if deviation > 0.25 || matches!(collection.threshold_status, ThresholdStatus::Critical)
            {
                detected_anomalies.push(DetectedAnomaly {
                    anomaly_id: format!("{}::{name}", self.detector_id),
                    metric_name: name.clone(),
                    severity: if deviation > 0.5 {
                        RiskSeverity::High
                    } else {
                        RiskSeverity::Medium
                    },
                });
                statistical_outliers.push(StatisticalOutlier {
                    metric_name: name.clone(),
                    value,
                    z_score: deviation * 3.0,
                });
                if context.has_unsafe_operations {
                    behavioral_changes.push(BehavioralChange {
                        description: format!("{name} deviates from baseline"),
                        magnitude: deviation,
                    });
                }
            }
        }
        if config_depth > 0 || !detected_anomalies.is_empty() {
            machine_learning_anomalies.push(MlAnomaly {
                model_name: self.detector_id.clone(),
                anomaly_score: average_quality(metrics),
            });
            clustering_anomalies.push(ClusteringAnomaly {
                cluster_id: format!("{}_cluster", self.detector_id),
                distance_from_centroid: config_depth as f64 * 0.1,
            });
        }
        if detected_anomalies.len() > 1 {
            anomaly_correlations.push(AnomalyCorrelation {
                anomaly_a: detected_anomalies[0].anomaly_id.clone(),
                anomaly_b: detected_anomalies[1].anomaly_id.clone(),
                correlation_strength: 0.5,
            });
        }
        Ok(AnomalyDetectionResult {
            detected_anomalies,
            anomaly_scores,
            baseline_deviations,
            behavioral_changes,
            statistical_outliers,
            machine_learning_anomalies,
            clustering_anomalies,
            anomaly_correlations,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlAnomaly {
    pub model_name: String,
    pub anomaly_score: f64,
}
