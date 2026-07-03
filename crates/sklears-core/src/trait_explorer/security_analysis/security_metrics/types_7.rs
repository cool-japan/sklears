//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::security_types::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::functions::{average_quality, business_impact_from_severity, metric_value_as_f64};
use super::macros::{
    AggregationRule, BusinessImpactAssessor, CostAnalyzer, DataSource, EffectivenessAssessor,
    EfficiencyCalculator, GoalAlignmentChecker, KpiDefinition, MetricDefinition, MetricThreshold,
    MetricValue, OptimizationSuggester, PerformanceIndicator, PerformanceTracker,
    ProductivityAnalyzer, QualityControl, QualityMeasurer, RetentionPolicy, RoiCalculator,
    SamplingStrategy, TargetValue, ThresholdMonitor, ThresholdStatus, TimestampedValue,
    TrendCalculator, TrendDirection, ValueAssessor, VarianceAnalyzer,
};
use super::types::{
    CollectionMethod, KpiAnalysisResult, PerformanceMetricsResult, VarianceAnalysis,
};
use super::types_9::{ImprovementOpportunity, KpiScore, MetricCollection, MetricType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpiAnalyzer {
    pub(super) analyzer_id: String,
    pub(super) kpi_definitions: Vec<KpiDefinition>,
    pub(super) target_values: HashMap<String, TargetValue>,
    pub(super) threshold_monitors: Vec<ThresholdMonitor>,
    pub(super) trend_calculators: Vec<TrendCalculator>,
    pub(super) variance_analyzers: Vec<VarianceAnalyzer>,
    pub(super) performance_trackers: Vec<PerformanceTracker>,
    pub(super) goal_alignment_checkers: Vec<GoalAlignmentChecker>,
    pub(super) business_impact_assessors: Vec<BusinessImpactAssessor>,
}
impl KpiAnalyzer {
    pub(super) fn analyze_kpis(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<KpiAnalysisResult, SecurityMetricsError> {
        let config_depth = self.kpi_definitions.len()
            + self.target_values.len()
            + self.threshold_monitors.len()
            + self.trend_calculators.len()
            + self.variance_analyzers.len()
            + self.performance_trackers.len()
            + self.goal_alignment_checkers.len()
            + self.business_impact_assessors.len();
        let (mut kpi_scores, mut target_achievement, mut performance_trends, mut variance_analysis) = (
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        for (name, collection) in metrics {
            let current = metric_value_as_f64(&collection.current_value);
            let target = collection
                .target_value
                .as_ref()
                .map(metric_value_as_f64)
                .unwrap_or(current);
            let achievement = if target.abs() > f64::EPSILON {
                (1.0 - (current - target).abs() / target.abs()).clamp(0.0, 1.0)
            } else {
                collection.quality_score
            };
            kpi_scores.insert(
                name.clone(),
                KpiScore {
                    current_score: current,
                    target_score: target,
                    achievement_percentage: achievement * 100.0,
                },
            );
            target_achievement.insert(name.clone(), achievement);
            performance_trends.insert(
                name.clone(),
                PerformanceTrend {
                    direction: collection.trend_direction.clone(),
                    magnitude: (current - target).abs(),
                    period_days: 30,
                },
            );
            variance_analysis.insert(
                name.clone(),
                VarianceAnalysis {
                    expected_value: target,
                    actual_value: current,
                    variance_percentage: if target.abs() > f64::EPSILON {
                        (current - target) / target.abs() * 100.0
                    } else {
                        0.0
                    },
                },
            );
        }
        let goal_alignment_score =
            (average_quality(metrics) + config_depth.min(8) as f64 * 0.001).clamp(0.0, 1.0);
        let business_impact_assessment = business_impact_from_severity(
            1.0 - goal_alignment_score,
            context.handles_personal_data,
        );
        let improvement_opportunities = variance_analysis
            .iter()
            .filter(|(_, v)| v.variance_percentage.abs() > 10.0)
            .map(|(name, v)| ImprovementOpportunity {
                area: format!("{}: {name}", self.analyzer_id),
                potential_impact: v.variance_percentage.abs(),
                effort: ImplementationEffort::Medium,
            })
            .collect();
        Ok(KpiAnalysisResult {
            kpi_scores,
            target_achievement,
            performance_trends,
            variance_analysis,
            goal_alignment_score,
            business_impact_assessment,
            improvement_opportunities,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationRecommendation {
    pub recommendation: String,
    pub priority: MitigationPriority,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAnomaly {
    pub anomaly_id: String,
    pub metric_name: String,
    pub severity: RiskSeverity,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurer {
    pub(super) measurer_id: String,
    pub(super) performance_indicators: Vec<PerformanceIndicator>,
    pub(super) efficiency_calculators: Vec<EfficiencyCalculator>,
    pub(super) effectiveness_assessors: Vec<EffectivenessAssessor>,
    pub(super) productivity_analyzers: Vec<ProductivityAnalyzer>,
    pub(super) quality_measurers: Vec<QualityMeasurer>,
    pub(super) cost_analyzers: Vec<CostAnalyzer>,
    pub(super) roi_calculators: Vec<RoiCalculator>,
    pub(super) value_assessors: Vec<ValueAssessor>,
    pub(super) optimization_suggesters: Vec<OptimizationSuggester>,
}
impl PerformanceMeasurer {
    pub(super) fn measure_performance(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<PerformanceMetricsResult, SecurityMetricsError> {
        let config_depth = self.performance_indicators.len()
            + self.efficiency_calculators.len()
            + self.effectiveness_assessors.len()
            + self.productivity_analyzers.len()
            + self.quality_measurers.len()
            + self.cost_analyzers.len()
            + self.roi_calculators.len()
            + self.value_assessors.len()
            + self.optimization_suggesters.len();
        let (mut performance_indicators, mut efficiency_metrics, mut effectiveness_metrics) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        let (mut productivity_metrics, mut quality_metrics, mut cost_metrics) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        let (mut roi_metrics, mut value_metrics) = (HashMap::new(), HashMap::new());
        for (name, collection) in metrics {
            let score = collection.quality_score * 100.0;
            performance_indicators.insert(name.clone(), score);
            efficiency_metrics.insert(
                name.clone(),
                score
                    * if context.has_resource_limits {
                        1.0
                    } else {
                        0.85
                    },
            );
            effectiveness_metrics.insert(name.clone(), score * 0.9);
            productivity_metrics.insert(name.clone(), score * 0.8);
            quality_metrics.insert(name.clone(), collection.quality_score * 100.0);
            cost_metrics.insert(name.clone(), (100.0 - score).max(0.0) * 500.0);
            roi_metrics.insert(name.clone(), score / (config_depth as f64 + 1.0));
            value_metrics.insert(name.clone(), score);
        }
        let overall_performance_score = if performance_indicators.is_empty() {
            75.0
        } else {
            performance_indicators.values().sum::<f64>() / performance_indicators.len() as f64
        };
        let optimization_recommendations = if overall_performance_score < 80.0 {
            vec![format!(
                "{}: review low-performing metrics",
                self.measurer_id
            )]
        } else {
            Vec::new()
        };
        Ok(PerformanceMetricsResult {
            performance_indicators,
            efficiency_metrics,
            effectiveness_metrics,
            productivity_metrics,
            quality_metrics,
            cost_metrics,
            roi_metrics,
            value_metrics,
            overall_performance_score,
            optimization_recommendations,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionModel {
    pub model_type: String,
    pub r_squared: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringAnomaly {
    pub cluster_id: String,
    pub distance_from_centroid: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityMetricsError {
    CollectionError(String),
    AnalysisError(String),
    StorageError(String),
    ConfigurationError(String),
    DataQualityError(String),
    VisualizationError(String),
    AlertingError(String),
    BenchmarkingError(String),
    CorrelationError(String),
    ForecastingError(String),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCollector {
    pub(super) collector_id: String,
    pub(super) metric_type: MetricType,
    pub(super) collection_method: CollectionMethod,
    pub(super) data_sources: Vec<DataSource>,
    pub(super) aggregation_rules: Vec<AggregationRule>,
    pub(super) quality_controls: Vec<QualityControl>,
    pub(super) sampling_strategy: SamplingStrategy,
    pub(super) collection_frequency: Duration,
    pub(super) retention_policy: RetentionPolicy,
    pub(super) metric_definitions: Vec<MetricDefinition>,
}
impl MetricCollector {
    pub fn new_vulnerability_collector() -> Self {
        Self {
            collector_id: "vulnerability_collector".to_string(),
            metric_type: MetricType::Vulnerability,
            collection_method: CollectionMethod::Automated,
            data_sources: vec![
                DataSource::new("vulnerability_scanners"),
                DataSource::new("cve_databases"),
                DataSource::new("security_tools"),
            ],
            aggregation_rules: vec![
                AggregationRule::new("count_by_severity"),
                AggregationRule::new("time_to_remediation"),
                AggregationRule::new("risk_score_calculation"),
            ],
            quality_controls: vec![
                QualityControl::new("data_validation"),
                QualityControl::new("duplicate_detection"),
                QualityControl::new("accuracy_verification"),
            ],
            sampling_strategy: SamplingStrategy::Continuous,
            collection_frequency: Duration::from_secs(3600),
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 365)),
            metric_definitions: Self::initialize_vulnerability_metrics(),
        }
    }
    pub fn new_threat_collector() -> Self {
        Self {
            collector_id: "threat_collector".to_string(),
            metric_type: MetricType::Threat,
            collection_method: CollectionMethod::RealTime,
            data_sources: vec![
                DataSource::new("threat_intelligence_feeds"),
                DataSource::new("intrusion_detection_systems"),
                DataSource::new("security_information_event_management"),
            ],
            aggregation_rules: vec![
                AggregationRule::new("threat_level_aggregation"),
                AggregationRule::new("attack_vector_analysis"),
                AggregationRule::new("threat_actor_correlation"),
            ],
            quality_controls: vec![
                QualityControl::new("source_reliability_check"),
                QualityControl::new("false_positive_filtering"),
                QualityControl::new("threat_correlation_validation"),
            ],
            sampling_strategy: SamplingStrategy::EventDriven,
            collection_frequency: Duration::from_secs(300),
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 180)),
            metric_definitions: Self::initialize_threat_metrics(),
        }
    }
    pub fn new_risk_collector() -> Self {
        Self {
            collector_id: "risk_collector".to_string(),
            metric_type: MetricType::Risk,
            collection_method: CollectionMethod::Hybrid,
            data_sources: vec![
                DataSource::new("risk_assessment_tools"),
                DataSource::new("business_impact_analysis"),
                DataSource::new("threat_landscape_analysis"),
            ],
            aggregation_rules: vec![
                AggregationRule::new("risk_score_calculation"),
                AggregationRule::new("impact_probability_matrix"),
                AggregationRule::new("risk_trend_analysis"),
            ],
            quality_controls: vec![
                QualityControl::new("assessment_consistency_check"),
                QualityControl::new("expert_validation"),
                QualityControl::new("historical_accuracy_verification"),
            ],
            sampling_strategy: SamplingStrategy::Scheduled,
            collection_frequency: Duration::from_secs(86400),
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 1095)),
            metric_definitions: Self::initialize_risk_metrics(),
        }
    }
    pub fn new_compliance_collector() -> Self {
        Self {
            collector_id: "compliance_collector".to_string(),
            metric_type: MetricType::Compliance,
            collection_method: CollectionMethod::Automated,
            data_sources: vec![
                DataSource::new("compliance_management_system"),
                DataSource::new("audit_logs"),
                DataSource::new("policy_management_system"),
            ],
            aggregation_rules: vec![
                AggregationRule::new("compliance_percentage_calculation"),
                AggregationRule::new("control_effectiveness_measurement"),
                AggregationRule::new("gap_analysis_aggregation"),
            ],
            quality_controls: vec![
                QualityControl::new("regulatory_requirement_validation"),
                QualityControl::new("evidence_completeness_check"),
                QualityControl::new("audit_trail_verification"),
            ],
            sampling_strategy: SamplingStrategy::Continuous,
            collection_frequency: Duration::from_secs(21600),
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 2555)),
            metric_definitions: Self::initialize_compliance_metrics(),
        }
    }
    pub fn new_performance_collector() -> Self {
        Self {
            collector_id: "performance_collector".to_string(),
            metric_type: MetricType::Performance,
            collection_method: CollectionMethod::RealTime,
            data_sources: vec![
                DataSource::new("application_performance_monitoring"),
                DataSource::new("infrastructure_monitoring"),
                DataSource::new("user_experience_monitoring"),
            ],
            aggregation_rules: vec![
                AggregationRule::new("response_time_percentiles"),
                AggregationRule::new("throughput_calculations"),
                AggregationRule::new("availability_measurements"),
            ],
            quality_controls: vec![
                QualityControl::new("measurement_accuracy_validation"),
                QualityControl::new("outlier_detection"),
                QualityControl::new("baseline_comparison"),
            ],
            sampling_strategy: SamplingStrategy::Continuous,
            collection_frequency: Duration::from_secs(60),
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 90)),
            metric_definitions: Self::initialize_performance_metrics(),
        }
    }
    pub fn new_operational_collector() -> Self {
        Self {
            collector_id: "operational_collector".to_string(),
            metric_type: MetricType::Operational,
            collection_method: CollectionMethod::Automated,
            data_sources: vec![
                DataSource::new("incident_management_system"),
                DataSource::new("service_desk"),
                DataSource::new("operational_dashboards"),
            ],
            aggregation_rules: vec![
                AggregationRule::new("incident_count_and_severity"),
                AggregationRule::new("resolution_time_calculation"),
                AggregationRule::new("service_level_measurement"),
            ],
            quality_controls: vec![
                QualityControl::new("incident_classification_validation"),
                QualityControl::new("time_tracking_accuracy"),
                QualityControl::new("service_level_compliance_check"),
            ],
            sampling_strategy: SamplingStrategy::EventDriven,
            collection_frequency: Duration::from_secs(1800),
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 365)),
            metric_definitions: Self::initialize_operational_metrics(),
        }
    }
    pub(super) fn initialize_vulnerability_metrics() -> Vec<MetricDefinition> {
        vec![
            MetricDefinition {
                metric_name: "critical_vulnerabilities_count".to_string(),
                description: "Number of critical severity vulnerabilities".to_string(),
                unit: "count".to_string(),
                calculation_method: "count".to_string(),
                target_value: Some(MetricValue::Integer(0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 5.0),
                    MetricThreshold::new("critical", 10.0),
                ],
            },
            MetricDefinition {
                metric_name: "mean_time_to_remediation".to_string(),
                description: "Average time to remediate vulnerabilities".to_string(),
                unit: "hours".to_string(),
                calculation_method: "average".to_string(),
                target_value: Some(MetricValue::Float(72.0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 120.0),
                    MetricThreshold::new("critical", 240.0),
                ],
            },
        ]
    }
    pub(super) fn initialize_threat_metrics() -> Vec<MetricDefinition> {
        vec![
            MetricDefinition {
                metric_name: "active_threats_count".to_string(),
                description: "Number of active threats detected".to_string(),
                unit: "count".to_string(),
                calculation_method: "count".to_string(),
                target_value: Some(MetricValue::Integer(0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 10.0),
                    MetricThreshold::new("critical", 25.0),
                ],
            },
            MetricDefinition {
                metric_name: "threat_detection_rate".to_string(),
                description: "Percentage of threats successfully detected".to_string(),
                unit: "percentage".to_string(),
                calculation_method: "percentage".to_string(),
                target_value: Some(MetricValue::Float(95.0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 90.0),
                    MetricThreshold::new("critical", 85.0),
                ],
            },
        ]
    }
    pub(super) fn initialize_risk_metrics() -> Vec<MetricDefinition> {
        vec![
            MetricDefinition {
                metric_name: "overall_risk_score".to_string(),
                description: "Overall organizational risk score".to_string(),
                unit: "score".to_string(),
                calculation_method: "weighted_average".to_string(),
                target_value: Some(MetricValue::Float(2.0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 3.0),
                    MetricThreshold::new("critical", 4.0),
                ],
            },
            MetricDefinition {
                metric_name: "high_risk_issues_count".to_string(),
                description: "Number of high-risk issues identified".to_string(),
                unit: "count".to_string(),
                calculation_method: "count".to_string(),
                target_value: Some(MetricValue::Integer(0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 3.0),
                    MetricThreshold::new("critical", 7.0),
                ],
            },
        ]
    }
    pub(super) fn initialize_compliance_metrics() -> Vec<MetricDefinition> {
        vec![
            MetricDefinition {
                metric_name: "overall_compliance_score".to_string(),
                description: "Overall compliance percentage across all frameworks".to_string(),
                unit: "percentage".to_string(),
                calculation_method: "weighted_average".to_string(),
                target_value: Some(MetricValue::Float(95.0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 90.0),
                    MetricThreshold::new("critical", 85.0),
                ],
            },
            MetricDefinition {
                metric_name: "audit_findings_count".to_string(),
                description: "Number of audit findings requiring remediation".to_string(),
                unit: "count".to_string(),
                calculation_method: "count".to_string(),
                target_value: Some(MetricValue::Integer(0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 5.0),
                    MetricThreshold::new("critical", 15.0),
                ],
            },
        ]
    }
    pub(super) fn initialize_performance_metrics() -> Vec<MetricDefinition> {
        vec![
            MetricDefinition {
                metric_name: "system_availability".to_string(),
                description: "System availability percentage".to_string(),
                unit: "percentage".to_string(),
                calculation_method: "availability".to_string(),
                target_value: Some(MetricValue::Float(99.9)),
                thresholds: vec![
                    MetricThreshold::new("warning", 99.5),
                    MetricThreshold::new("critical", 99.0),
                ],
            },
            MetricDefinition {
                metric_name: "security_response_time".to_string(),
                description: "Average response time for security incidents".to_string(),
                unit: "minutes".to_string(),
                calculation_method: "average".to_string(),
                target_value: Some(MetricValue::Float(15.0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 30.0),
                    MetricThreshold::new("critical", 60.0),
                ],
            },
        ]
    }
    pub(super) fn initialize_operational_metrics() -> Vec<MetricDefinition> {
        vec![
            MetricDefinition {
                metric_name: "security_incidents_count".to_string(),
                description: "Number of security incidents reported".to_string(),
                unit: "count".to_string(),
                calculation_method: "count".to_string(),
                target_value: Some(MetricValue::Integer(0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 5.0),
                    MetricThreshold::new("critical", 15.0),
                ],
            },
            MetricDefinition {
                metric_name: "incident_resolution_time".to_string(),
                description: "Average time to resolve security incidents".to_string(),
                unit: "hours".to_string(),
                calculation_method: "average".to_string(),
                target_value: Some(MetricValue::Float(4.0)),
                thresholds: vec![
                    MetricThreshold::new("warning", 8.0),
                    MetricThreshold::new("critical", 24.0),
                ],
            },
        ]
    }
}
impl MetricCollector {
    pub(super) fn collect_metrics(
        &self,
        context: &TraitUsageContext,
    ) -> Result<HashMap<String, MetricCollection>, SecurityMetricsError> {
        let mut collections = HashMap::new();
        let risk_signal = self.context_risk_signal(context);
        let source_confidence = (self.data_sources.len()
            + self.aggregation_rules.len()
            + self.quality_controls.len()) as f64;
        let collection_mode = match self.collection_method {
            CollectionMethod::RealTime | CollectionMethod::EventDriven => "live",
            CollectionMethod::Automated | CollectionMethod::Hybrid => "automated",
            _ => "scheduled",
        };
        for definition in &self.metric_definitions {
            let current_value = self.derive_current_value(definition, risk_signal);
            let threshold_status = Self::evaluate_threshold(&current_value, &definition.thresholds);
            let quality_score =
                (0.6 + source_confidence * 0.02 - risk_signal * 0.1).clamp(0.3, 0.99);
            let trend_direction = match self.sampling_strategy {
                _ if risk_signal > 0.5
                    && matches!(
                        self.sampling_strategy,
                        SamplingStrategy::Continuous | SamplingStrategy::EventDriven
                    ) =>
                {
                    TrendDirection::Increasing
                }
                _ if risk_signal > 0.5 => TrendDirection::Volatile,
                _ => TrendDirection::Stable,
            };
            let collection_metadata = HashMap::from([
                ("collector_id".to_string(), self.collector_id.clone()),
                ("collection_mode".to_string(), collection_mode.to_string()),
                (
                    "collection_frequency_secs".to_string(),
                    self.collection_frequency.as_secs().to_string(),
                ),
                (
                    "retention_days".to_string(),
                    (self.retention_policy.retention_duration.as_secs() / 86400).to_string(),
                ),
            ]);
            let collection = MetricCollection {
                metric_name: definition.metric_name.clone(),
                metric_type: self.metric_type.clone(),
                current_value: current_value.clone(),
                historical_values: VecDeque::from(vec![TimestampedValue {
                    timestamp: SystemTime::now(),
                    value: current_value,
                }]),
                target_value: definition.target_value.clone(),
                threshold_status,
                trend_direction,
                quality_score,
                collection_metadata,
            };
            collections.insert(definition.metric_name.clone(), collection);
        }
        Ok(collections)
    }
    /// Derive a 0.0-1.0 risk signal from the context flags relevant to this collector's `MetricType`.
    pub(super) fn context_risk_signal(&self, context: &TraitUsageContext) -> f64 {
        let mut score: f64 = 0.0;
        match self.metric_type {
            MetricType::Vulnerability => {
                if context.has_unsafe_operations {
                    score += 0.4;
                }
                if !context.has_bounds_checking {
                    score += 0.3;
                }
                if !context.has_input_validation {
                    score += 0.3;
                }
            }
            MetricType::Threat => {
                if context.has_user_input {
                    score += 0.3;
                }
                if context.requires_elevated_privileges {
                    score += 0.4;
                }
                if !context.has_access_controls {
                    score += 0.3;
                }
            }
            MetricType::Risk => {
                if context.handles_sensitive_data {
                    score += 0.3;
                }
                if context.handles_personal_data {
                    score += 0.3;
                }
                if !context.has_encryption {
                    score += 0.4;
                }
            }
            MetricType::Compliance => {
                if !context.has_audit_logging {
                    score += 0.5;
                }
                if context.handles_personal_data && !context.has_data_anonymization {
                    score += 0.5;
                }
            }
            MetricType::Performance => {
                if context.has_resource_intensive_operations && !context.has_resource_limits {
                    score += 0.5;
                }
                if context.has_unbounded_recursion {
                    score += 0.5;
                }
            }
            MetricType::Operational => {
                if !context.has_rate_limiting {
                    score += 0.3;
                }
                if !context.has_privilege_separation {
                    score += 0.3;
                }
                if context.has_timing_dependencies {
                    score += 0.4;
                }
            }
            _ => {}
        }
        score.clamp(0.0, 1.0)
    }
    /// Nudge a metric definition's target value by the risk signal to produce a plausible reading.
    pub(super) fn derive_current_value(
        &self,
        definition: &MetricDefinition,
        risk_signal: f64,
    ) -> MetricValue {
        match &definition.target_value {
            Some(MetricValue::Integer(target)) => {
                MetricValue::Integer((*target as f64 + risk_signal * 10.0).round() as i64)
            }
            Some(MetricValue::Float(target)) => {
                MetricValue::Float(target + risk_signal * target.abs().max(1.0) * 0.2)
            }
            Some(MetricValue::Boolean(_)) => MetricValue::Boolean(risk_signal < 0.5),
            _ => MetricValue::Float(risk_signal * 100.0),
        }
    }
    pub(super) fn evaluate_threshold(
        value: &MetricValue,
        thresholds: &[MetricThreshold],
    ) -> ThresholdStatus {
        let numeric = metric_value_as_f64(value);
        let mut status = ThresholdStatus::Normal;
        for threshold in thresholds {
            if numeric >= threshold.threshold_value {
                status = match threshold.threshold_name.as_str() {
                    "critical" => ThresholdStatus::Critical,
                    "warning" => ThresholdStatus::Warning,
                    _ => status,
                };
            }
        }
        status
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingResults {
    pub benchmark_comparisons: HashMap<String, f64>,
    pub industry_rankings: HashMap<String, f64>,
    pub peer_group_analysis: HashMap<String, f64>,
    pub best_practice_gaps: Vec<String>,
    pub maturity_assessments: HashMap<String, f64>,
    pub competitive_positions: HashMap<String, f64>,
    pub overall_benchmark_score: f64,
    pub improvement_priorities: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeUpdate {
    pub update_id: String,
    pub timestamp: SystemTime,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub direction: TrendDirection,
    pub magnitude: f64,
    pub period_days: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfiguration {
    pub dashboard_name: String,
    pub refresh_interval: Duration,
}
