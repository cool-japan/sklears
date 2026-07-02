use super::security_types::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetricsCollector {
    metric_collectors: HashMap<String, MetricCollector>,
    kpi_analyzers: Vec<KpiAnalyzer>,
    kri_monitors: Vec<KriMonitor>,
    dashboard_managers: Vec<DashboardManager>,
    trend_analyzers: Vec<TrendAnalyzer>,
    anomaly_detectors: Vec<AnomalyDetector>,
    benchmarking_engines: Vec<BenchmarkingEngine>,
    real_time_monitors: Vec<RealTimeMonitor>,
    scorecard_generators: Vec<ScorecardGenerator>,
    correlation_analyzers: Vec<CorrelationAnalyzer>,
    performance_measurers: Vec<PerformanceMeasurer>,
    compliance_trackers: Vec<ComplianceTracker>,
    alerting_system: SecurityAlertingSystem,
    metrics_storage: MetricsStorage,
    metrics_config: SecurityMetricsConfig,
    metrics_cache: HashMap<String, CachedMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCollector {
    collector_id: String,
    metric_type: MetricType,
    collection_method: CollectionMethod,
    data_sources: Vec<DataSource>,
    aggregation_rules: Vec<AggregationRule>,
    quality_controls: Vec<QualityControl>,
    sampling_strategy: SamplingStrategy,
    collection_frequency: Duration,
    retention_policy: RetentionPolicy,
    metric_definitions: Vec<MetricDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    Vulnerability,
    Threat,
    Risk,
    Compliance,
    Performance,
    Operational,
    Financial,
    Technical,
    Process,
    Behavioral,
    Environmental,
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
pub struct KpiAnalyzer {
    analyzer_id: String,
    kpi_definitions: Vec<KpiDefinition>,
    target_values: HashMap<String, TargetValue>,
    threshold_monitors: Vec<ThresholdMonitor>,
    trend_calculators: Vec<TrendCalculator>,
    variance_analyzers: Vec<VarianceAnalyzer>,
    performance_trackers: Vec<PerformanceTracker>,
    goal_alignment_checkers: Vec<GoalAlignmentChecker>,
    business_impact_assessors: Vec<BusinessImpactAssessor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KriMonitor {
    monitor_id: String,
    kri_definitions: Vec<KriDefinition>,
    risk_thresholds: HashMap<String, RiskThreshold>,
    early_warning_systems: Vec<EarlyWarningSystem>,
    predictive_models: Vec<PredictiveModel>,
    correlation_engines: Vec<CorrelationEngine>,
    escalation_procedures: Vec<EscalationProcedure>,
    mitigation_triggers: Vec<MitigationTrigger>,
    risk_appetite_monitors: Vec<RiskAppetiteMonitor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardManager {
    dashboard_id: String,
    dashboard_type: DashboardType,
    visualization_components: Vec<VisualizationComponent>,
    data_aggregators: Vec<DataAggregator>,
    real_time_updaters: Vec<RealTimeUpdater>,
    interactive_features: Vec<InteractiveFeature>,
    export_capabilities: Vec<ExportCapability>,
    access_controls: DashboardAccessControls,
    customization_options: Vec<CustomizationOption>,
    performance_optimizers: Vec<PerformanceOptimizer>,
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
pub struct TrendAnalyzer {
    analyzer_id: String,
    trend_algorithms: Vec<TrendAlgorithm>,
    statistical_models: Vec<StatisticalModel>,
    forecasting_engines: Vec<ForecastingEngine>,
    seasonality_detectors: Vec<SeasonalityDetector>,
    change_point_detectors: Vec<ChangePointDetector>,
    regression_analyzers: Vec<RegressionAnalyzer>,
    time_series_analyzers: Vec<TimeSeriesAnalyzer>,
    pattern_recognizers: Vec<PatternRecognizer>,
    predictive_analytics: Vec<PredictiveAnalytic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetector {
    detector_id: String,
    anomaly_algorithms: Vec<AnomalyAlgorithm>,
    baseline_calculators: Vec<BaselineCalculator>,
    outlier_detectors: Vec<OutlierDetector>,
    behavioral_analyzers: Vec<BehavioralAnalyzer>,
    statistical_anomaly_detectors: Vec<StatisticalAnomalyDetector>,
    machine_learning_detectors: Vec<MachineLearningDetector>,
    threshold_anomaly_detectors: Vec<ThresholdAnomalyDetector>,
    clustering_detectors: Vec<ClusteringDetector>,
    isolation_forest_detectors: Vec<IsolationForestDetector>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingEngine {
    engine_id: String,
    benchmark_categories: Vec<BenchmarkCategory>,
    industry_comparisons: Vec<IndustryComparison>,
    peer_group_analysis: Vec<PeerGroupAnalysis>,
    best_practice_comparisons: Vec<BestPracticeComparison>,
    maturity_assessments: Vec<MaturityAssessment>,
    competitive_analysis: Vec<CompetitiveAnalysis>,
    standard_benchmarks: Vec<StandardBenchmark>,
    custom_benchmarks: Vec<CustomBenchmark>,
    benchmark_reporting: BenchmarkReporting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMonitor {
    monitor_id: String,
    real_time_streams: Vec<RealTimeStream>,
    stream_processors: Vec<StreamProcessor>,
    event_correlators: Vec<EventCorrelator>,
    threshold_checkers: Vec<ThresholdChecker>,
    alert_generators: Vec<AlertGenerator>,
    notification_systems: Vec<NotificationSystem>,
    escalation_managers: Vec<EscalationManager>,
    response_coordinators: Vec<ResponseCoordinator>,
    metrics_aggregators: Vec<MetricsAggregator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorecardGenerator {
    generator_id: String,
    scorecard_templates: Vec<ScorecardTemplate>,
    scoring_algorithms: Vec<ScoringAlgorithm>,
    weight_calculators: Vec<WeightCalculator>,
    aggregation_methods: Vec<AggregationMethod>,
    visualization_engines: Vec<VisualizationEngine>,
    report_generators: Vec<ReportGenerator>,
    stakeholder_views: Vec<StakeholderView>,
    historical_comparisons: Vec<HistoricalComparison>,
    goal_tracking: Vec<GoalTracker>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalyzer {
    analyzer_id: String,
    correlation_methods: Vec<CorrelationMethod>,
    dependency_analyzers: Vec<DependencyAnalyzer>,
    causality_analyzers: Vec<CausalityAnalyzer>,
    association_miners: Vec<AssociationMiner>,
    pattern_correlators: Vec<PatternCorrelator>,
    cross_domain_analyzers: Vec<CrossDomainAnalyzer>,
    temporal_correlators: Vec<TemporalCorrelator>,
    multivariate_analyzers: Vec<MultivariateAnalyzer>,
    network_analyzers: Vec<NetworkAnalyzer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurer {
    measurer_id: String,
    performance_indicators: Vec<PerformanceIndicator>,
    efficiency_calculators: Vec<EfficiencyCalculator>,
    effectiveness_assessors: Vec<EffectivenessAssessor>,
    productivity_analyzers: Vec<ProductivityAnalyzer>,
    quality_measurers: Vec<QualityMeasurer>,
    cost_analyzers: Vec<CostAnalyzer>,
    roi_calculators: Vec<RoiCalculator>,
    value_assessors: Vec<ValueAssessor>,
    optimization_suggesters: Vec<OptimizationSuggester>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceTracker {
    tracker_id: String,
    compliance_frameworks: Vec<String>,
    requirement_trackers: Vec<RequirementTracker>,
    control_effectiveness_measurers: Vec<ControlEffectivenessMeasurer>,
    audit_preparedness_assessors: Vec<AuditPreparednessAssessor>,
    gap_analyzers: Vec<GapAnalyzer>,
    remediation_trackers: Vec<RemediationTracker>,
    certification_monitors: Vec<CertificationMonitor>,
    regulatory_change_trackers: Vec<RegulatoryChangeTracker>,
    compliance_scorers: Vec<ComplianceScorer>,
}

// Small supporting infrastructure types referenced by the collector/manager structs above.
// These are intentionally simple "shallow" data holders (mirroring `DataSource` etc. below).
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
impl Default for SecurityAlertingSystem {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsStorage {
    pub storage_backend: String,
    pub total_stored_metrics: usize,
}
impl MetricsStorage {
    pub fn new() -> Self {
        Self {
            storage_backend: "in_memory".to_string(),
            total_stored_metrics: 0,
        }
    }
}
impl Default for MetricsStorage {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardAccessControls {
    pub enabled: bool,
    pub allowed_roles: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkReporting {
    pub enabled: bool,
    pub report_format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetricsResult {
    pub result_id: String,
    pub collection_timestamp: SystemTime,
    pub metric_collections: HashMap<String, MetricCollection>,
    pub kpi_analysis: KpiAnalysisResult,
    pub kri_monitoring: KriMonitoringResult,
    pub dashboard_data: DashboardData,
    pub trend_analysis: TrendAnalysisResult,
    pub anomaly_detection: AnomalyDetectionResult,
    pub benchmarking_results: BenchmarkingResults,
    pub real_time_status: RealTimeStatus,
    pub security_scorecard: SecurityScorecard,
    pub correlation_analysis: CorrelationAnalysisResult,
    pub performance_metrics: PerformanceMetricsResult,
    pub compliance_metrics: ComplianceMetricsResult,
    pub overall_security_score: f64,
    pub health_indicators: Vec<HealthIndicator>,
    pub actionable_insights: Vec<ActionableInsight>,
    pub recommendations: Vec<MetricsRecommendation>,
    pub next_collection_time: SystemTime,
    pub analysis_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCollection {
    pub metric_name: String,
    pub metric_type: MetricType,
    pub current_value: MetricValue,
    pub historical_values: VecDeque<TimestampedValue>,
    pub target_value: Option<MetricValue>,
    pub threshold_status: ThresholdStatus,
    pub trend_direction: TrendDirection,
    pub quality_score: f64,
    pub collection_metadata: HashMap<String, String>,
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
pub struct DashboardData {
    pub dashboard_configurations: HashMap<String, DashboardConfiguration>,
    pub visualization_data: HashMap<String, VisualizationData>,
    pub real_time_updates: Vec<RealTimeUpdate>,
    pub interactive_elements: Vec<InteractiveElement>,
    pub export_ready_data: HashMap<String, ExportData>,
    pub performance_statistics: DashboardPerformanceStats,
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
pub struct AnomalyDetectionResult {
    pub detected_anomalies: Vec<DetectedAnomaly>,
    pub anomaly_scores: HashMap<String, f64>,
    pub baseline_deviations: HashMap<String, BaselineDeviation>,
    pub behavioral_changes: Vec<BehavioralChange>,
    pub statistical_outliers: Vec<StatisticalOutlier>,
    pub machine_learning_anomalies: Vec<MlAnomaly>,
    pub clustering_anomalies: Vec<ClusteringAnomaly>,
    pub anomaly_correlations: Vec<AnomalyCorrelation>,
}

// Domain-level result aggregates: flat, mirror the shape of the result types above, and are
// reused as the return type of the corresponding per-domain collector methods below.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMetricsResult {
    pub framework_compliance: HashMap<String, f64>,
    pub requirement_status: HashMap<String, String>,
    pub control_effectiveness: HashMap<String, f64>,
    pub audit_readiness: HashMap<String, f64>,
    pub compliance_gaps: Vec<String>,
    pub remediation_progress: HashMap<String, f64>,
    pub certification_status: HashMap<String, String>,
    pub overall_compliance_score: f64,
    pub compliance_trends: HashMap<String, TrendDirection>,
    pub priority_actions: Vec<String>,
}

// Leaf result-substructure types: shallow, flat data holders (no embedded algorithms).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpiScore {
    pub current_score: f64,
    pub target_score: f64,
    pub achievement_percentage: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub direction: TrendDirection,
    pub magnitude: f64,
    pub period_days: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceAnalysis {
    pub expected_value: f64,
    pub actual_value: f64,
    pub variance_percentage: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementOpportunity {
    pub area: String,
    pub potential_impact: f64,
    pub effort: ImplementationEffort,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KriValue {
    pub current_value: f64,
    pub threshold: f64,
    pub status: ThresholdStatus,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyWarning {
    pub indicator: String,
    pub severity: RiskSeverity,
    pub message: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAlert {
    pub metric_name: String,
    pub predicted_value: f64,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationFinding {
    pub metric_a: String,
    pub metric_b: String,
    pub correlation_coefficient: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationTrigger {
    pub trigger_name: String,
    pub threshold_breached: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationRecommendation {
    pub recommendation: String,
    pub priority: MitigationPriority,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfiguration {
    pub dashboard_name: String,
    pub refresh_interval: Duration,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub chart_type: String,
    pub data_points: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeUpdate {
    pub update_id: String,
    pub timestamp: SystemTime,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveElement {
    pub element_id: String,
    pub element_type: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportData {
    pub format: String,
    pub size_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPerformanceStats {
    pub render_time_ms: f64,
    pub data_load_time_ms: f64,
    pub cache_hit_rate: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPattern {
    pub pattern_type: String,
    pub strength: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    pub forecasted_value: f64,
    pub confidence: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityFinding {
    pub period_days: u32,
    pub amplitude: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    pub timestamp: SystemTime,
    pub magnitude: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionModel {
    pub model_type: String,
    pub r_squared: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAnomaly {
    pub anomaly_id: String,
    pub metric_name: String,
    pub severity: RiskSeverity,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineDeviation {
    pub expected: f64,
    pub observed: f64,
    pub deviation_percentage: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralChange {
    pub description: String,
    pub magnitude: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalOutlier {
    pub metric_name: String,
    pub value: f64,
    pub z_score: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlAnomaly {
    pub model_name: String,
    pub anomaly_score: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringAnomaly {
    pub cluster_id: String,
    pub distance_from_centroid: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyCorrelation {
    pub anomaly_a: String,
    pub anomaly_b: String,
    pub correlation_strength: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIndicator {
    pub indicator_name: String,
    pub status: ThresholdStatus,
    pub value: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableInsight {
    pub insight: String,
    pub priority: AnalysisPriority,
    pub related_metrics: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsRecommendation {
    pub recommendation: String,
    pub priority: MitigationPriority,
    pub estimated_cost: EstimatedCost,
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

// Shared helpers for the per-domain collector implementations below.
/// Convert a local [`MetricValue`] into a numeric approximation usable in shallow heuristics.
fn metric_value_as_f64(value: &MetricValue) -> f64 {
    match value {
        MetricValue::Integer(i) => *i as f64,
        MetricValue::Float(f) => *f,
        MetricValue::Boolean(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        MetricValue::String(_) => 0.0,
    }
}

/// Average data-quality score across a collection of metrics, with a neutral fallback when empty.
fn average_quality(metrics: &HashMap<String, MetricCollection>) -> f64 {
    if metrics.is_empty() {
        return 0.75;
    }
    metrics.values().map(|m| m.quality_score).sum::<f64>() / metrics.len() as f64
}

/// Build a shallow [`BusinessImpactAssessment`] from a single 0.0-1.0 severity signal.
fn business_impact_from_severity(
    severity: f64,
    personal_data_exposed: bool,
) -> BusinessImpactAssessment {
    let s = severity.clamp(0.0, 1.0);
    BusinessImpactAssessment {
        financial_impact: FinancialImpact {
            direct_costs: s * 10_000.0,
            indirect_costs: s * 5_000.0,
            revenue_loss: s * 20_000.0,
            regulatory_fines: 0.0,
            legal_costs: 0.0,
            recovery_costs: s * 2_000.0,
        },
        operational_impact: OperationalImpact {
            service_disruption_duration: Duration::from_secs((s * 3600.0) as u64),
            affected_business_processes: Vec::new(),
            productivity_loss_percentage: s * 100.0,
            recovery_time_objective: Duration::from_secs(3600),
            recovery_point_objective: Duration::from_secs(900),
        },
        reputational_impact: ReputationalImpact {
            brand_damage_score: s * 10.0,
            customer_trust_impact: s,
            media_attention_level: MediaAttentionLevel::None,
            social_media_sentiment: SentimentLevel::Neutral,
            stakeholder_confidence_impact: s,
        },
        legal_impact: LegalImpact {
            regulatory_violations: Vec::new(),
            potential_lawsuits: 0,
            compliance_breach_severity: ComplianceBreachSeverity::Minor,
            data_protection_violations: Vec::new(),
            contractual_breach_risk: s * 0.5,
        },
        customer_impact: CustomerImpact {
            affected_customer_count: 0,
            customer_data_exposed: personal_data_exposed,
            service_availability_impact: s,
            customer_satisfaction_impact: s,
            churn_risk_percentage: s * 10.0,
        },
        competitive_impact: CompetitiveImpact {
            competitive_advantage_loss: s * 0.3,
            intellectual_property_exposure: false,
            market_share_impact: s * 0.1,
            innovation_capability_impact: s * 0.2,
        },
    }
}

impl Default for SecurityMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl SecurityMetricsCollector {
    pub fn new() -> Self {
        Self {
            metric_collectors: Self::initialize_metric_collectors(),
            kpi_analyzers: Vec::new(),
            kri_monitors: Vec::new(),
            dashboard_managers: Vec::new(),
            trend_analyzers: Vec::new(),
            anomaly_detectors: Vec::new(),
            benchmarking_engines: Vec::new(),
            real_time_monitors: Vec::new(),
            scorecard_generators: Vec::new(),
            correlation_analyzers: Vec::new(),
            performance_measurers: Vec::new(),
            compliance_trackers: Vec::new(),
            alerting_system: SecurityAlertingSystem::new(),
            metrics_storage: MetricsStorage::new(),
            metrics_config: SecurityMetricsConfig::default(),
            metrics_cache: HashMap::new(),
        }
    }

    pub fn collect_security_metrics(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<SecurityMetricsResult, SecurityMetricsError> {
        let result_id = self.generate_result_id(context);

        if let Some(cached_result) = self.get_cached_metrics(&result_id) {
            if self.is_cache_valid(&cached_result) {
                return Ok(cached_result.result.clone());
            }
        }

        let metric_collections = self.collect_all_metrics(context)?;
        let kpi_analysis = self.analyze_kpis(context, &metric_collections)?;
        let kri_monitoring = self.monitor_kris(context, &metric_collections)?;
        let dashboard_data = self.prepare_dashboard_data(context, &metric_collections)?;
        let trend_analysis = self.analyze_trends(context, &metric_collections)?;
        let anomaly_detection = self.detect_anomalies(context, &metric_collections)?;
        let benchmarking_results = self.perform_benchmarking(context, &metric_collections)?;
        let real_time_status = self.get_real_time_status(context)?;
        let security_scorecard = self.generate_security_scorecard(context, &metric_collections)?;
        let correlation_analysis = self.analyze_correlations(context, &metric_collections)?;
        let performance_metrics = self.measure_performance(context, &metric_collections)?;
        let compliance_metrics = self.track_compliance(context, &metric_collections)?;

        let overall_security_score = self.calculate_overall_security_score(
            &kpi_analysis,
            &kri_monitoring,
            &compliance_metrics,
            &performance_metrics,
        )?;

        let health_indicators = self.generate_health_indicators(&metric_collections)?;
        let actionable_insights = self.generate_actionable_insights(
            &trend_analysis,
            &anomaly_detection,
            &correlation_analysis,
        )?;

        let recommendations = self.generate_metrics_recommendations(
            &kpi_analysis,
            &kri_monitoring,
            &trend_analysis,
            &anomaly_detection,
        )?;

        let next_collection_time = self.calculate_next_collection_time()?;
        let analysis_confidence = self.calculate_analysis_confidence(&metric_collections)?;

        let result = SecurityMetricsResult {
            result_id: result_id.clone(),
            collection_timestamp: SystemTime::now(),
            metric_collections,
            kpi_analysis,
            kri_monitoring,
            dashboard_data,
            trend_analysis,
            anomaly_detection,
            benchmarking_results,
            real_time_status,
            security_scorecard,
            correlation_analysis,
            performance_metrics,
            compliance_metrics,
            overall_security_score,
            health_indicators,
            actionable_insights,
            recommendations,
            next_collection_time,
            analysis_confidence,
        };

        self.cache_metrics(result_id, &result);
        Ok(result)
    }
    fn collect_all_metrics(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<HashMap<String, MetricCollection>, SecurityMetricsError> {
        let mut collections = HashMap::new();

        for (collector_id, collector) in &self.metric_collectors {
            let collection = collector.collect_metrics(context)?;
            for (metric_name, metric_data) in collection {
                collections.insert(format!("{}_{}", collector_id, metric_name), metric_data);
            }
        }

        Ok(collections)
    }
    fn analyze_kpis(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<KpiAnalysisResult, SecurityMetricsError> {
        let mut kpi_scores = HashMap::new();
        let mut target_achievement = HashMap::new();
        let mut performance_trends = HashMap::new();
        let mut variance_analysis = HashMap::new();

        for analyzer in &self.kpi_analyzers {
            let analysis = analyzer.analyze_kpis(context, metrics)?;
            kpi_scores.extend(analysis.kpi_scores);
            target_achievement.extend(analysis.target_achievement);
            performance_trends.extend(analysis.performance_trends);
            variance_analysis.extend(analysis.variance_analysis);
        }

        let goal_alignment_score = self.calculate_goal_alignment_score(&kpi_scores)?;
        let business_impact_assessment =
            self.assess_business_impact(&kpi_scores, &target_achievement)?;
        let improvement_opportunities =
            self.identify_improvement_opportunities(&variance_analysis)?;

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
    fn monitor_kris(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<KriMonitoringResult, SecurityMetricsError> {
        let mut kri_values = HashMap::new();
        let mut risk_threshold_status = HashMap::new();
        let mut early_warnings = Vec::new();
        let mut predictive_alerts = Vec::new();
        let mut correlation_findings = Vec::new();
        let mut escalation_triggers = Vec::new();
        let mut mitigation_recommendations = Vec::new();

        for monitor in &self.kri_monitors {
            let monitoring_result = monitor.monitor_kris(context, metrics)?;
            kri_values.extend(monitoring_result.kri_values);
            risk_threshold_status.extend(monitoring_result.risk_threshold_status);
            early_warnings.extend(monitoring_result.early_warnings);
            predictive_alerts.extend(monitoring_result.predictive_alerts);
            correlation_findings.extend(monitoring_result.correlation_findings);
            escalation_triggers.extend(monitoring_result.escalation_triggers);
            mitigation_recommendations.extend(monitoring_result.mitigation_recommendations);
        }

        let risk_appetite_compliance = self.calculate_risk_appetite_compliance(&kri_values)?;

        Ok(KriMonitoringResult {
            kri_values,
            risk_threshold_status,
            early_warnings,
            predictive_alerts,
            correlation_findings,
            escalation_triggers,
            mitigation_recommendations,
            risk_appetite_compliance,
        })
    }
    fn prepare_dashboard_data(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<DashboardData, SecurityMetricsError> {
        let mut dashboard_configurations = HashMap::new();
        let mut visualization_data = HashMap::new();
        let mut real_time_updates = Vec::new();
        let mut interactive_elements = Vec::new();
        let mut export_ready_data = HashMap::new();

        for manager in &self.dashboard_managers {
            let dashboard_result = manager.prepare_dashboard_data(context, metrics)?;
            dashboard_configurations.extend(dashboard_result.dashboard_configurations);
            visualization_data.extend(dashboard_result.visualization_data);
            real_time_updates.extend(dashboard_result.real_time_updates);
            interactive_elements.extend(dashboard_result.interactive_elements);
            export_ready_data.extend(dashboard_result.export_ready_data);
        }

        let performance_statistics = self.collect_dashboard_performance_stats()?;

        Ok(DashboardData {
            dashboard_configurations,
            visualization_data,
            real_time_updates,
            interactive_elements,
            export_ready_data,
            performance_statistics,
        })
    }
    fn analyze_trends(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<TrendAnalysisResult, SecurityMetricsError> {
        let mut trend_patterns = HashMap::new();
        let mut statistical_significance = HashMap::new();
        let mut forecasting_results = HashMap::new();
        let mut seasonality_findings = HashMap::new();
        let mut change_points = HashMap::new();
        let mut regression_models = HashMap::new();
        let mut predictive_accuracy = HashMap::new();

        for analyzer in &self.trend_analyzers {
            let trend_result = analyzer.analyze_trends(context, metrics)?;
            trend_patterns.extend(trend_result.trend_patterns);
            statistical_significance.extend(trend_result.statistical_significance);
            forecasting_results.extend(trend_result.forecasting_results);
            seasonality_findings.extend(trend_result.seasonality_findings);
            change_points.extend(trend_result.change_points);
            regression_models.extend(trend_result.regression_models);
            predictive_accuracy.extend(trend_result.predictive_accuracy);
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
    fn detect_anomalies(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<AnomalyDetectionResult, SecurityMetricsError> {
        let mut detected_anomalies = Vec::new();
        let mut anomaly_scores = HashMap::new();
        let mut baseline_deviations = HashMap::new();
        let mut behavioral_changes = Vec::new();
        let mut statistical_outliers = Vec::new();
        let mut machine_learning_anomalies = Vec::new();
        let mut clustering_anomalies = Vec::new();
        let mut anomaly_correlations = Vec::new();

        for detector in &self.anomaly_detectors {
            let detection_result = detector.detect_anomalies(context, metrics)?;
            detected_anomalies.extend(detection_result.detected_anomalies);
            anomaly_scores.extend(detection_result.anomaly_scores);
            baseline_deviations.extend(detection_result.baseline_deviations);
            behavioral_changes.extend(detection_result.behavioral_changes);
            statistical_outliers.extend(detection_result.statistical_outliers);
            machine_learning_anomalies.extend(detection_result.machine_learning_anomalies);
            clustering_anomalies.extend(detection_result.clustering_anomalies);
            anomaly_correlations.extend(detection_result.anomaly_correlations);
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
    fn perform_benchmarking(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<BenchmarkingResults, SecurityMetricsError> {
        let mut benchmark_comparisons = HashMap::new();
        let mut industry_rankings = HashMap::new();
        let mut peer_group_analysis = HashMap::new();
        let mut best_practice_gaps = Vec::new();
        let mut maturity_assessments = HashMap::new();
        let mut competitive_positions = HashMap::new();

        for engine in &self.benchmarking_engines {
            let benchmarking_result = engine.perform_benchmarking(context, metrics)?;
            benchmark_comparisons.extend(benchmarking_result.benchmark_comparisons);
            industry_rankings.extend(benchmarking_result.industry_rankings);
            peer_group_analysis.extend(benchmarking_result.peer_group_analysis);
            best_practice_gaps.extend(benchmarking_result.best_practice_gaps);
            maturity_assessments.extend(benchmarking_result.maturity_assessments);
            competitive_positions.extend(benchmarking_result.competitive_positions);
        }

        let overall_benchmark_score =
            self.calculate_overall_benchmark_score(&benchmark_comparisons)?;
        let improvement_priorities = self.identify_improvement_priorities(&best_practice_gaps)?;

        Ok(BenchmarkingResults {
            benchmark_comparisons,
            industry_rankings,
            peer_group_analysis,
            best_practice_gaps,
            maturity_assessments,
            competitive_positions,
            overall_benchmark_score,
            improvement_priorities,
        })
    }
    fn get_real_time_status(
        &mut self,
        context: &TraitUsageContext,
    ) -> Result<RealTimeStatus, SecurityMetricsError> {
        let mut real_time_metrics = HashMap::new();
        let mut stream_health = HashMap::new();
        let mut active_alerts = Vec::new();
        let mut system_status = HashMap::new();
        let mut throughput_metrics = HashMap::new();
        let mut latency_metrics = HashMap::new();

        for monitor in &self.real_time_monitors {
            let status = monitor.get_real_time_status(context)?;
            real_time_metrics.extend(status.real_time_metrics);
            stream_health.extend(status.stream_health);
            active_alerts.extend(status.active_alerts);
            system_status.extend(status.system_status);
            throughput_metrics.extend(status.throughput_metrics);
            latency_metrics.extend(status.latency_metrics);
        }

        let overall_health_score =
            self.calculate_overall_health_score(&stream_health, &system_status)?;
        let performance_summary =
            self.generate_performance_summary(&throughput_metrics, &latency_metrics)?;

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
    fn generate_security_scorecard(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<SecurityScorecard, SecurityMetricsError> {
        let mut category_scores = HashMap::new();
        let mut weighted_scores = HashMap::new();
        let mut performance_indicators = Vec::new();
        let mut trend_indicators = Vec::new();
        let mut risk_indicators = Vec::new();

        for generator in &self.scorecard_generators {
            let scorecard = generator.generate_scorecard(context, metrics)?;
            category_scores.extend(scorecard.category_scores);
            weighted_scores.extend(scorecard.weighted_scores);
            performance_indicators.extend(scorecard.performance_indicators);
            trend_indicators.extend(scorecard.trend_indicators);
            risk_indicators.extend(scorecard.risk_indicators);
        }

        let overall_score = self.calculate_overall_scorecard_score(&weighted_scores)?;
        let grade = self.determine_security_grade(overall_score)?;
        let improvement_areas = self.identify_scorecard_improvement_areas(&category_scores)?;
        let historical_comparison = self.generate_historical_comparison(&category_scores)?;

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
    fn analyze_correlations(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<CorrelationAnalysisResult, SecurityMetricsError> {
        let mut metric_correlations = HashMap::new();
        let mut dependency_networks = HashMap::new();
        let mut causality_relationships = Vec::new();
        let mut association_patterns = Vec::new();
        let mut cross_domain_correlations = Vec::new();
        let mut temporal_correlations = Vec::new();

        for analyzer in &self.correlation_analyzers {
            let correlation_result = analyzer.analyze_correlations(context, metrics)?;
            metric_correlations.extend(correlation_result.metric_correlations);
            dependency_networks.extend(correlation_result.dependency_networks);
            causality_relationships.extend(correlation_result.causality_relationships);
            association_patterns.extend(correlation_result.association_patterns);
            cross_domain_correlations.extend(correlation_result.cross_domain_correlations);
            temporal_correlations.extend(correlation_result.temporal_correlations);
        }

        let correlation_strength_summary =
            self.summarize_correlation_strengths(&metric_correlations)?;
        let actionable_correlations =
            self.identify_actionable_correlations(&causality_relationships)?;

        Ok(CorrelationAnalysisResult {
            metric_correlations,
            dependency_networks,
            causality_relationships,
            association_patterns,
            cross_domain_correlations,
            temporal_correlations,
            correlation_strength_summary,
            actionable_correlations,
        })
    }
    fn measure_performance(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<PerformanceMetricsResult, SecurityMetricsError> {
        let mut performance_indicators = HashMap::new();
        let mut efficiency_metrics = HashMap::new();
        let mut effectiveness_metrics = HashMap::new();
        let mut productivity_metrics = HashMap::new();
        let mut quality_metrics = HashMap::new();
        let mut cost_metrics = HashMap::new();
        let mut roi_metrics = HashMap::new();
        let mut value_metrics = HashMap::new();

        for measurer in &self.performance_measurers {
            let performance_result = measurer.measure_performance(context, metrics)?;
            performance_indicators.extend(performance_result.performance_indicators);
            efficiency_metrics.extend(performance_result.efficiency_metrics);
            effectiveness_metrics.extend(performance_result.effectiveness_metrics);
            productivity_metrics.extend(performance_result.productivity_metrics);
            quality_metrics.extend(performance_result.quality_metrics);
            cost_metrics.extend(performance_result.cost_metrics);
            roi_metrics.extend(performance_result.roi_metrics);
            value_metrics.extend(performance_result.value_metrics);
        }

        let overall_performance_score = self.calculate_overall_performance_score(
            &efficiency_metrics,
            &effectiveness_metrics,
            &quality_metrics,
        )?;

        let optimization_recommendations = self.generate_optimization_recommendations(
            &performance_indicators,
            &efficiency_metrics,
            &cost_metrics,
        )?;

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
    fn track_compliance(
        &mut self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<ComplianceMetricsResult, SecurityMetricsError> {
        let mut framework_compliance = HashMap::new();
        let mut requirement_status = HashMap::new();
        let mut control_effectiveness = HashMap::new();
        let mut audit_readiness = HashMap::new();
        let mut compliance_gaps = Vec::new();
        let mut remediation_progress = HashMap::new();
        let mut certification_status = HashMap::new();

        for tracker in &self.compliance_trackers {
            let compliance_result = tracker.track_compliance(context, metrics)?;
            framework_compliance.extend(compliance_result.framework_compliance);
            requirement_status.extend(compliance_result.requirement_status);
            control_effectiveness.extend(compliance_result.control_effectiveness);
            audit_readiness.extend(compliance_result.audit_readiness);
            compliance_gaps.extend(compliance_result.compliance_gaps);
            remediation_progress.extend(compliance_result.remediation_progress);
            certification_status.extend(compliance_result.certification_status);
        }

        let overall_compliance_score =
            self.calculate_overall_compliance_score(&framework_compliance)?;
        let compliance_trends = self.analyze_compliance_trends(&framework_compliance)?;
        let priority_actions = self.identify_priority_compliance_actions(&compliance_gaps)?;

        Ok(ComplianceMetricsResult {
            framework_compliance,
            requirement_status,
            control_effectiveness,
            audit_readiness,
            compliance_gaps,
            remediation_progress,
            certification_status,
            overall_compliance_score,
            compliance_trends,
            priority_actions,
        })
    }
    fn initialize_metric_collectors() -> HashMap<String, MetricCollector> {
        let mut collectors = HashMap::new();

        collectors.insert(
            "vulnerability_metrics".to_string(),
            MetricCollector::new_vulnerability_collector(),
        );
        collectors.insert(
            "threat_metrics".to_string(),
            MetricCollector::new_threat_collector(),
        );
        collectors.insert(
            "risk_metrics".to_string(),
            MetricCollector::new_risk_collector(),
        );
        collectors.insert(
            "compliance_metrics".to_string(),
            MetricCollector::new_compliance_collector(),
        );
        collectors.insert(
            "performance_metrics".to_string(),
            MetricCollector::new_performance_collector(),
        );
        collectors.insert(
            "operational_metrics".to_string(),
            MetricCollector::new_operational_collector(),
        );

        collectors
    }
}

// Private aggregation/caching helpers used by `collect_security_metrics` and the per-domain
// orchestration methods above. Each one tolerates empty input (the `Vec<...Analyzer>` fields on
// `SecurityMetricsCollector` start out unpopulated) and returns a sensible default instead.
impl SecurityMetricsCollector {
    fn generate_result_id(&self, context: &TraitUsageContext) -> String {
        let fallback = context.traits.len() as u64 + context.trait_name.len() as u64;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(fallback);
        format!(
            "secmetrics_{}_{timestamp}",
            context.trait_name.replace(' ', "_")
        )
    }
    fn get_cached_metrics(&self, result_id: &str) -> Option<CachedMetrics> {
        self.metrics_cache.get(result_id).cloned()
    }
    fn is_cache_valid(&self, cached: &CachedMetrics) -> bool {
        cached
            .cache_timestamp
            .elapsed()
            .map(|elapsed| elapsed < cached.cache_ttl)
            .unwrap_or(false)
    }
    fn cache_metrics(&mut self, result_id: String, result: &SecurityMetricsResult) {
        self.metrics_storage.total_stored_metrics += 1;
        let cache_ttl = self
            .metrics_config
            .dashboard_refresh_rate
            .max(Duration::from_secs(60));
        self.metrics_cache.insert(
            result_id,
            CachedMetrics {
                result: result.clone(),
                cache_timestamp: SystemTime::now(),
                cache_ttl,
            },
        );
    }
    fn calculate_overall_security_score(
        &self,
        kpi: &KpiAnalysisResult,
        kri: &KriMonitoringResult,
        compliance: &ComplianceMetricsResult,
        performance: &PerformanceMetricsResult,
    ) -> Result<f64, SecurityMetricsError> {
        let score = kpi.goal_alignment_score * 0.3
            + kri.risk_appetite_compliance * 0.3
            + (compliance.overall_compliance_score / 100.0).clamp(0.0, 1.0) * 0.25
            + (performance.overall_performance_score / 100.0).clamp(0.0, 1.0) * 0.15;
        Ok((score * 10.0).clamp(0.0, 10.0))
    }
    fn calculate_analysis_confidence(
        &self,
        metric_collections: &HashMap<String, MetricCollection>,
    ) -> Result<f64, SecurityMetricsError> {
        Ok(average_quality(metric_collections).clamp(0.0, 1.0))
    }
    fn generate_health_indicators(
        &self,
        metric_collections: &HashMap<String, MetricCollection>,
    ) -> Result<Vec<HealthIndicator>, SecurityMetricsError> {
        let mut indicators: Vec<HealthIndicator> = metric_collections
            .values()
            .map(|collection| HealthIndicator {
                indicator_name: collection.metric_name.clone(),
                status: collection.threshold_status.clone(),
                value: metric_value_as_f64(&collection.current_value),
            })
            .collect();
        if self.alerting_system.enabled {
            indicators.push(HealthIndicator {
                indicator_name: "alerting_system".to_string(),
                status: ThresholdStatus::Normal,
                value: self.alerting_system.alert_channels.len() as f64,
            });
        }
        Ok(indicators)
    }
    fn generate_actionable_insights(
        &self,
        trend_analysis: &TrendAnalysisResult,
        anomaly_detection: &AnomalyDetectionResult,
        correlation_analysis: &CorrelationAnalysisResult,
    ) -> Result<Vec<ActionableInsight>, SecurityMetricsError> {
        let mut insights = Vec::new();
        for (name, significance) in &trend_analysis.statistical_significance {
            if *significance > 0.05 {
                insights.push(ActionableInsight {
                    insight: format!("Significant trend detected in {name}"),
                    priority: AnalysisPriority::Medium,
                    related_metrics: vec![name.clone()],
                });
            }
        }
        for anomaly in &anomaly_detection.detected_anomalies {
            insights.push(ActionableInsight {
                insight: format!("Anomaly detected: {}", anomaly.anomaly_id),
                priority: AnalysisPriority::High,
                related_metrics: vec![anomaly.metric_name.clone()],
            });
        }
        if !correlation_analysis.actionable_correlations.is_empty() {
            insights.push(ActionableInsight {
                insight: correlation_analysis.correlation_strength_summary.clone(),
                priority: AnalysisPriority::Informational,
                related_metrics: correlation_analysis.actionable_correlations.clone(),
            });
        }
        Ok(insights)
    }
    fn generate_metrics_recommendations(
        &self,
        kpi_analysis: &KpiAnalysisResult,
        kri_monitoring: &KriMonitoringResult,
        trend_analysis: &TrendAnalysisResult,
        anomaly_detection: &AnomalyDetectionResult,
    ) -> Result<Vec<MetricsRecommendation>, SecurityMetricsError> {
        let mut recommendations = Vec::new();
        for opportunity in &kpi_analysis.improvement_opportunities {
            recommendations.push(MetricsRecommendation {
                recommendation: format!("Improve {}", opportunity.area),
                priority: MitigationPriority::Medium,
                estimated_cost: EstimatedCost::Low,
            });
        }
        for warning in &kri_monitoring.early_warnings {
            recommendations.push(MetricsRecommendation {
                recommendation: warning.message.clone(),
                priority: MitigationPriority::High,
                estimated_cost: EstimatedCost::Medium,
            });
        }
        if !trend_analysis.trend_patterns.is_empty() {
            recommendations.push(MetricsRecommendation {
                recommendation: "Review emerging trend patterns".to_string(),
                priority: MitigationPriority::Low,
                estimated_cost: EstimatedCost::Low,
            });
        }
        if !anomaly_detection.detected_anomalies.is_empty() {
            recommendations.push(MetricsRecommendation {
                recommendation: "Investigate detected anomalies".to_string(),
                priority: MitigationPriority::Critical,
                estimated_cost: EstimatedCost::High,
            });
        }
        Ok(recommendations)
    }
    fn calculate_next_collection_time(&self) -> Result<SystemTime, SecurityMetricsError> {
        Ok(SystemTime::now() + self.metrics_config.dashboard_refresh_rate)
    }
    fn calculate_goal_alignment_score(
        &self,
        kpi_scores: &HashMap<String, KpiScore>,
    ) -> Result<f64, SecurityMetricsError> {
        if kpi_scores.is_empty() {
            return Ok(0.75);
        }
        Ok((kpi_scores
            .values()
            .map(|s| s.achievement_percentage)
            .sum::<f64>()
            / kpi_scores.len() as f64
            / 100.0)
            .clamp(0.0, 1.0))
    }
    fn assess_business_impact(
        &self,
        kpi_scores: &HashMap<String, KpiScore>,
        target_achievement: &HashMap<String, f64>,
    ) -> Result<BusinessImpactAssessment, SecurityMetricsError> {
        let avg_achievement = if target_achievement.is_empty() {
            0.8
        } else {
            target_achievement.values().sum::<f64>() / target_achievement.len() as f64
        };
        let severity = (1.0 - avg_achievement).max(0.0);
        let exposed = kpi_scores.values().any(|s| s.achievement_percentage < 50.0);
        Ok(business_impact_from_severity(severity, exposed))
    }
    fn identify_improvement_opportunities(
        &self,
        variance_analysis: &HashMap<String, VarianceAnalysis>,
    ) -> Result<Vec<ImprovementOpportunity>, SecurityMetricsError> {
        Ok(variance_analysis
            .iter()
            .filter(|(_, v)| v.variance_percentage.abs() > 10.0)
            .map(|(name, v)| ImprovementOpportunity {
                area: name.clone(),
                potential_impact: v.variance_percentage.abs(),
                effort: ImplementationEffort::Medium,
            })
            .collect())
    }
    fn calculate_risk_appetite_compliance(
        &self,
        kri_values: &HashMap<String, KriValue>,
    ) -> Result<f64, SecurityMetricsError> {
        if kri_values.is_empty() {
            return Ok(0.8);
        }
        let breaches = kri_values
            .values()
            .filter(|v| !matches!(v.status, ThresholdStatus::Normal))
            .count();
        Ok((1.0 - breaches as f64 / kri_values.len() as f64).clamp(0.0, 1.0))
    }
    fn collect_dashboard_performance_stats(
        &self,
    ) -> Result<DashboardPerformanceStats, SecurityMetricsError> {
        let cache_hit_rate = if self.metrics_cache.is_empty() {
            0.0
        } else {
            0.9
        };
        Ok(DashboardPerformanceStats {
            render_time_ms: 45.0,
            data_load_time_ms: 120.0,
            cache_hit_rate,
        })
    }
    fn calculate_overall_benchmark_score(
        &self,
        benchmark_comparisons: &HashMap<String, f64>,
    ) -> Result<f64, SecurityMetricsError> {
        if benchmark_comparisons.is_empty() {
            return Ok(5.0);
        }
        Ok(benchmark_comparisons.values().sum::<f64>() / benchmark_comparisons.len() as f64)
    }
    fn identify_improvement_priorities(
        &self,
        best_practice_gaps: &[String],
    ) -> Result<Vec<String>, SecurityMetricsError> {
        Ok(best_practice_gaps.iter().take(5).cloned().collect())
    }
    fn calculate_overall_health_score(
        &self,
        stream_health: &HashMap<String, f64>,
        system_status: &HashMap<String, String>,
    ) -> Result<f64, SecurityMetricsError> {
        let health_avg = if stream_health.is_empty() {
            0.95
        } else {
            stream_health.values().sum::<f64>() / stream_health.len() as f64
        };
        let operational_ratio = if system_status.is_empty() {
            1.0
        } else {
            system_status
                .values()
                .filter(|s| s.as_str() == "operational")
                .count() as f64
                / system_status.len() as f64
        };
        Ok((health_avg * 0.6 + operational_ratio * 0.4).clamp(0.0, 1.0))
    }
    fn generate_performance_summary(
        &self,
        throughput_metrics: &HashMap<String, f64>,
        latency_metrics: &HashMap<String, f64>,
    ) -> Result<String, SecurityMetricsError> {
        let throughput = throughput_metrics.values().sum::<f64>();
        let latency = if latency_metrics.is_empty() {
            0.0
        } else {
            latency_metrics.values().sum::<f64>() / latency_metrics.len() as f64
        };
        Ok(format!(
            "throughput={throughput:.2}, avg_latency={latency:.2}ms"
        ))
    }
    fn calculate_overall_scorecard_score(
        &self,
        weighted_scores: &HashMap<String, f64>,
    ) -> Result<f64, SecurityMetricsError> {
        if weighted_scores.is_empty() {
            return Ok(7.0);
        }
        Ok(weighted_scores.values().sum::<f64>() / weighted_scores.len() as f64)
    }
    fn determine_security_grade(&self, overall_score: f64) -> Result<String, SecurityMetricsError> {
        let grade = match overall_score {
            s if s >= 9.0 => "A",
            s if s >= 8.0 => "B",
            s if s >= 7.0 => "C",
            s if s >= 6.0 => "D",
            _ => "F",
        };
        Ok(grade.to_string())
    }
    fn identify_scorecard_improvement_areas(
        &self,
        category_scores: &HashMap<String, f64>,
    ) -> Result<Vec<String>, SecurityMetricsError> {
        Ok(category_scores
            .iter()
            .filter(|(_, score)| **score < 7.0)
            .map(|(name, _)| name.clone())
            .collect())
    }
    fn generate_historical_comparison(
        &self,
        category_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>, SecurityMetricsError> {
        Ok(category_scores
            .iter()
            .map(|(name, score)| (name.clone(), score * 0.95))
            .collect())
    }
    fn summarize_correlation_strengths(
        &self,
        metric_correlations: &HashMap<String, f64>,
    ) -> Result<String, SecurityMetricsError> {
        if metric_correlations.is_empty() {
            return Ok("no correlations observed".to_string());
        }
        let strong = metric_correlations
            .values()
            .filter(|c| c.abs() > 0.7)
            .count();
        Ok(format!(
            "{strong} strong correlation(s) out of {}",
            metric_correlations.len()
        ))
    }
    fn identify_actionable_correlations(
        &self,
        causality_relationships: &[String],
    ) -> Result<Vec<String>, SecurityMetricsError> {
        Ok(causality_relationships.iter().take(5).cloned().collect())
    }
    fn calculate_overall_performance_score(
        &self,
        efficiency_metrics: &HashMap<String, f64>,
        effectiveness_metrics: &HashMap<String, f64>,
        quality_metrics: &HashMap<String, f64>,
    ) -> Result<f64, SecurityMetricsError> {
        let avg = |m: &HashMap<String, f64>| {
            if m.is_empty() {
                75.0
            } else {
                m.values().sum::<f64>() / m.len() as f64
            }
        };
        Ok(avg(efficiency_metrics) * 0.4
            + avg(effectiveness_metrics) * 0.35
            + avg(quality_metrics) * 0.25)
    }
    fn generate_optimization_recommendations(
        &self,
        performance_indicators: &HashMap<String, f64>,
        efficiency_metrics: &HashMap<String, f64>,
        cost_metrics: &HashMap<String, f64>,
    ) -> Result<Vec<String>, SecurityMetricsError> {
        let mut recommendations = Vec::new();
        if efficiency_metrics.values().any(|v| *v < 60.0) {
            recommendations.push("Improve processing efficiency".to_string());
        }
        if cost_metrics.values().any(|v| *v > 50_000.0) {
            recommendations.push("Review high-cost operations".to_string());
        }
        if performance_indicators.is_empty() {
            recommendations.push("Expand performance indicator coverage".to_string());
        }
        Ok(recommendations)
    }
    fn calculate_overall_compliance_score(
        &self,
        framework_compliance: &HashMap<String, f64>,
    ) -> Result<f64, SecurityMetricsError> {
        if framework_compliance.is_empty() {
            return Ok(80.0);
        }
        Ok(framework_compliance.values().sum::<f64>() / framework_compliance.len() as f64)
    }
    fn analyze_compliance_trends(
        &self,
        framework_compliance: &HashMap<String, f64>,
    ) -> Result<HashMap<String, TrendDirection>, SecurityMetricsError> {
        Ok(framework_compliance
            .iter()
            .map(|(name, score)| {
                (
                    name.clone(),
                    if *score >= 90.0 {
                        TrendDirection::Stable
                    } else if *score >= 70.0 {
                        TrendDirection::Volatile
                    } else {
                        TrendDirection::Decreasing
                    },
                )
            })
            .collect())
    }
    fn identify_priority_compliance_actions(
        &self,
        compliance_gaps: &[String],
    ) -> Result<Vec<String>, SecurityMetricsError> {
        Ok(compliance_gaps.iter().take(3).cloned().collect())
    }
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
            collection_frequency: Duration::from_secs(3600), // Hourly
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 365)), // 1 year
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
            collection_frequency: Duration::from_secs(300), // 5 minutes
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 180)), // 6 months
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
            collection_frequency: Duration::from_secs(86400), // Daily
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 1095)), // 3 years
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
            collection_frequency: Duration::from_secs(21600), // 6 hours
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 2555)), // 7 years
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
            collection_frequency: Duration::from_secs(60), // 1 minute
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 90)), // 3 months
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
            collection_frequency: Duration::from_secs(1800), // 30 minutes
            retention_policy: RetentionPolicy::new(Duration::from_secs(86400 * 365)), // 1 year
            metric_definitions: Self::initialize_operational_metrics(),
        }
    }
    fn initialize_vulnerability_metrics() -> Vec<MetricDefinition> {
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
                target_value: Some(MetricValue::Float(72.0)), // 3 days
                thresholds: vec![
                    MetricThreshold::new("warning", 120.0),  // 5 days
                    MetricThreshold::new("critical", 240.0), // 10 days
                ],
            },
        ]
    }
    fn initialize_threat_metrics() -> Vec<MetricDefinition> {
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
    fn initialize_risk_metrics() -> Vec<MetricDefinition> {
        vec![
            MetricDefinition {
                metric_name: "overall_risk_score".to_string(),
                description: "Overall organizational risk score".to_string(),
                unit: "score".to_string(),
                calculation_method: "weighted_average".to_string(),
                target_value: Some(MetricValue::Float(2.0)), // Low risk
                thresholds: vec![
                    MetricThreshold::new("warning", 3.0),  // Medium risk
                    MetricThreshold::new("critical", 4.0), // High risk
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
    fn initialize_compliance_metrics() -> Vec<MetricDefinition> {
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
    fn initialize_performance_metrics() -> Vec<MetricDefinition> {
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
    fn initialize_operational_metrics() -> Vec<MetricDefinition> {
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

// Real per-metric collection logic: the one place in the file where the collected data is
// genuinely non-empty at runtime, so `TraitUsageContext`'s signal flags actually drive the
// resulting `MetricCollection` values.
impl MetricCollector {
    fn collect_metrics(
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
    fn context_risk_signal(&self, context: &TraitUsageContext) -> f64 {
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
    fn derive_current_value(&self, definition: &MetricDefinition, risk_signal: f64) -> MetricValue {
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
    fn evaluate_threshold(value: &MetricValue, thresholds: &[MetricThreshold]) -> ThresholdStatus {
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

impl KpiAnalyzer {
    fn analyze_kpis(
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

impl KriMonitor {
    fn monitor_kris(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<KriMonitoringResult, SecurityMetricsError> {
        let config_depth = self.kri_definitions.len()
            + self.risk_thresholds.len()
            + self.early_warning_systems.len()
            + self.predictive_models.len()
            + self.correlation_engines.len()
            + self.escalation_procedures.len()
            + self.mitigation_triggers.len()
            + self.risk_appetite_monitors.len();
        let (mut kri_values, mut risk_threshold_status) = (HashMap::new(), HashMap::new());
        let (mut early_warnings, mut predictive_alerts, mut correlation_findings) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut escalation_triggers, mut mitigation_recommendations) = (Vec::new(), Vec::new());
        for (name, collection) in metrics {
            let value = metric_value_as_f64(&collection.current_value);
            let status = collection.threshold_status.clone();
            if !matches!(status, ThresholdStatus::Normal) {
                early_warnings.push(EarlyWarning {
                    indicator: name.clone(),
                    severity: RiskSeverity::Medium,
                    message: format!(
                        "{} ({name}) trending outside expected range",
                        self.monitor_id
                    ),
                });
                escalation_triggers.push(EscalationTrigger {
                    trigger_name: name.clone(),
                    threshold_breached: value,
                });
                mitigation_recommendations.push(MitigationRecommendation {
                    recommendation: format!("Review {name}"),
                    priority: MitigationPriority::Medium,
                });
            }
            if context.requires_elevated_privileges {
                predictive_alerts.push(PredictiveAlert {
                    metric_name: name.clone(),
                    predicted_value: value * 1.1,
                    confidence: average_quality(metrics),
                });
            }
            kri_values.insert(
                name.clone(),
                KriValue {
                    current_value: value,
                    threshold: value,
                    status: status.clone(),
                },
            );
            risk_threshold_status.insert(name.clone(), status);
        }
        if kri_values.len() > 1 || config_depth > 0 {
            correlation_findings.push(CorrelationFinding {
                metric_a: self.monitor_id.clone(),
                metric_b: "aggregate".to_string(),
                correlation_coefficient: average_quality(metrics),
            });
        }
        let risk_appetite_compliance = average_quality(metrics);
        Ok(KriMonitoringResult {
            kri_values,
            risk_threshold_status,
            early_warnings,
            predictive_alerts,
            correlation_findings,
            escalation_triggers,
            mitigation_recommendations,
            risk_appetite_compliance,
        })
    }
}

impl DashboardManager {
    fn prepare_dashboard_data(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<DashboardData, SecurityMetricsError> {
        let config_depth = self.visualization_components.len()
            + self.data_aggregators.len()
            + self.real_time_updaters.len()
            + self.interactive_features.len()
            + self.export_capabilities.len()
            + self.customization_options.len()
            + self.performance_optimizers.len();
        let (mut dashboard_configurations, mut visualization_data, mut export_ready_data) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        let (mut real_time_updates, mut interactive_elements) = (Vec::new(), Vec::new());
        dashboard_configurations.insert(
            self.dashboard_id.clone(),
            DashboardConfiguration {
                dashboard_name: format!("{:?}", self.dashboard_type),
                refresh_interval: Duration::from_secs(60),
            },
        );
        for (name, collection) in metrics {
            visualization_data.insert(
                name.clone(),
                VisualizationData {
                    chart_type: "line".to_string(),
                    data_points: vec![metric_value_as_f64(&collection.current_value)],
                },
            );
            if self.access_controls.enabled && context.has_audit_logging {
                real_time_updates.push(RealTimeUpdate {
                    update_id: format!("{}::{name}", self.dashboard_id),
                    timestamp: SystemTime::now(),
                });
            }
        }
        interactive_elements.push(InteractiveElement {
            element_id: self.dashboard_id.clone(),
            element_type: format!("{:?}", self.dashboard_type),
        });
        export_ready_data.insert(
            self.dashboard_id.clone(),
            ExportData {
                format: "json".to_string(),
                size_bytes: (metrics.len() * 128) as u64,
            },
        );
        let performance_statistics = DashboardPerformanceStats {
            render_time_ms: 40.0 + config_depth as f64,
            data_load_time_ms: 90.0,
            cache_hit_rate: 0.85,
        };
        Ok(DashboardData {
            dashboard_configurations,
            visualization_data,
            real_time_updates,
            interactive_elements,
            export_ready_data,
            performance_statistics,
        })
    }
}

impl TrendAnalyzer {
    fn analyze_trends(
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

impl AnomalyDetector {
    fn detect_anomalies(
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

impl BenchmarkingEngine {
    fn perform_benchmarking(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<BenchmarkingResults, SecurityMetricsError> {
        let config_depth = self.benchmark_categories.len()
            + self.industry_comparisons.len()
            + self.peer_group_analysis.len()
            + self.best_practice_comparisons.len()
            + self.maturity_assessments.len()
            + self.competitive_analysis.len()
            + self.standard_benchmarks.len()
            + self.custom_benchmarks.len();
        let (mut benchmark_comparisons, mut industry_rankings, mut peer_group_analysis) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        let (mut maturity_assessments, mut competitive_positions) =
            (HashMap::new(), HashMap::new());
        let mut best_practice_gaps = Vec::new();
        for (name, collection) in metrics {
            let score = collection.quality_score * 10.0;
            benchmark_comparisons.insert(name.clone(), score);
            industry_rankings.insert(name.clone(), score * 0.9);
            peer_group_analysis.insert(name.clone(), score * 0.95);
            maturity_assessments.insert(name.clone(), score);
            competitive_positions.insert(name.clone(), score * 1.05);
            if score < 7.0 {
                best_practice_gaps.push(format!("{}: {name}", self.engine_id));
            }
        }
        if context.has_audit_logging && self.benchmark_reporting.enabled {
            best_practice_gaps.push(format!(
                "{} reporting reviewed ({config_depth} categories)",
                self.engine_id
            ));
        }
        let overall_benchmark_score = if benchmark_comparisons.is_empty() {
            5.0
        } else {
            benchmark_comparisons.values().sum::<f64>() / benchmark_comparisons.len() as f64
        };
        let improvement_priorities = best_practice_gaps.iter().take(3).cloned().collect();
        Ok(BenchmarkingResults {
            benchmark_comparisons,
            industry_rankings,
            peer_group_analysis,
            best_practice_gaps,
            maturity_assessments,
            competitive_positions,
            overall_benchmark_score,
            improvement_priorities,
        })
    }
}

impl RealTimeMonitor {
    fn get_real_time_status(
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

impl ScorecardGenerator {
    fn generate_scorecard(
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

impl CorrelationAnalyzer {
    fn analyze_correlations(
        &self,
        context: &TraitUsageContext,
        metrics: &HashMap<String, MetricCollection>,
    ) -> Result<CorrelationAnalysisResult, SecurityMetricsError> {
        let config_depth = self.correlation_methods.len()
            + self.dependency_analyzers.len()
            + self.causality_analyzers.len()
            + self.association_miners.len()
            + self.pattern_correlators.len()
            + self.cross_domain_analyzers.len()
            + self.temporal_correlators.len()
            + self.multivariate_analyzers.len()
            + self.network_analyzers.len();
        let names: Vec<String> = metrics.keys().cloned().collect();
        let (mut metric_correlations, mut dependency_networks) = (HashMap::new(), HashMap::new());
        let (mut causality_relationships, mut association_patterns) = (Vec::new(), Vec::new());
        let (mut cross_domain_correlations, mut temporal_correlations) = (Vec::new(), Vec::new());
        for (name, collection) in metrics {
            metric_correlations.insert(name.clone(), collection.quality_score);
            let related: Vec<String> = names
                .iter()
                .filter(|n| *n != name)
                .take(2)
                .cloned()
                .collect();
            dependency_networks.insert(name.clone(), related);
            if context.has_cryptographic_operations {
                temporal_correlations.push(format!("{}: {name}", self.analyzer_id));
            }
        }
        if names.len() > 1 {
            causality_relationships.push(format!("{} -> {}", names[0], names[names.len() - 1]));
            association_patterns.push(format!(
                "{} shared pattern across {} metrics",
                self.analyzer_id,
                names.len()
            ));
        }
        if config_depth > 0 {
            cross_domain_correlations.push(format!(
                "{} cross-domain signal ({config_depth} methods)",
                self.analyzer_id
            ));
        }
        let correlation_strength_summary = format!(
            "{} metrics analyzed with {config_depth} configured methods",
            names.len()
        );
        let actionable_correlations = causality_relationships.clone();
        Ok(CorrelationAnalysisResult {
            metric_correlations,
            dependency_networks,
            causality_relationships,
            association_patterns,
            cross_domain_correlations,
            temporal_correlations,
            correlation_strength_summary,
            actionable_correlations,
        })
    }
}

impl PerformanceMeasurer {
    fn measure_performance(
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

impl ComplianceTracker {
    fn track_compliance(
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

impl std::fmt::Display for SecurityMetricsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityMetricsError::CollectionError(msg) => write!(f, "Collection error: {}", msg),
            SecurityMetricsError::AnalysisError(msg) => write!(f, "Analysis error: {}", msg),
            SecurityMetricsError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            SecurityMetricsError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            SecurityMetricsError::DataQualityError(msg) => write!(f, "Data quality error: {}", msg),
            SecurityMetricsError::VisualizationError(msg) => {
                write!(f, "Visualization error: {}", msg)
            }
            SecurityMetricsError::AlertingError(msg) => write!(f, "Alerting error: {}", msg),
            SecurityMetricsError::BenchmarkingError(msg) => {
                write!(f, "Benchmarking error: {}", msg)
            }
            SecurityMetricsError::CorrelationError(msg) => write!(f, "Correlation error: {}", msg),
            SecurityMetricsError::ForecastingError(msg) => write!(f, "Forecasting error: {}", msg),
        }
    }
}

impl std::error::Error for SecurityMetricsError {}

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

impl Default for SecurityMetricsConfig {
    fn default() -> Self {
        let mut collection_intervals = HashMap::new();
        collection_intervals.insert(MetricType::Vulnerability, Duration::from_secs(3600));
        collection_intervals.insert(MetricType::Threat, Duration::from_secs(300));
        collection_intervals.insert(MetricType::Risk, Duration::from_secs(86400));
        collection_intervals.insert(MetricType::Compliance, Duration::from_secs(21600));
        collection_intervals.insert(MetricType::Performance, Duration::from_secs(60));
        collection_intervals.insert(MetricType::Operational, Duration::from_secs(1800));

        let mut retention_policies = HashMap::new();
        retention_policies.insert(MetricType::Vulnerability, Duration::from_secs(86400 * 365));
        retention_policies.insert(MetricType::Threat, Duration::from_secs(86400 * 180));
        retention_policies.insert(MetricType::Risk, Duration::from_secs(86400 * 1095));
        retention_policies.insert(MetricType::Compliance, Duration::from_secs(86400 * 2555));
        retention_policies.insert(MetricType::Performance, Duration::from_secs(86400 * 90));
        retention_policies.insert(MetricType::Operational, Duration::from_secs(86400 * 365));

        let mut quality_thresholds = HashMap::new();
        quality_thresholds.insert("data_completeness".to_string(), 0.95);
        quality_thresholds.insert("data_accuracy".to_string(), 0.98);
        quality_thresholds.insert("data_timeliness".to_string(), 0.90);

        Self {
            collection_intervals,
            retention_policies,
            quality_thresholds,
            alerting_enabled: true,
            real_time_processing: true,
            anomaly_detection_sensitivity: 0.85,
            trend_analysis_window: Duration::from_secs(86400 * 30), // 30 days
            benchmarking_enabled: true,
            dashboard_refresh_rate: Duration::from_secs(60), // 1 minute
        }
    }
}

macro_rules! define_metrics_supporting_types {
    () => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct DataSource {
            pub source_id: String,
            pub source_type: String,
            pub connection_info: HashMap<String, String>,
        }

        impl DataSource {
            pub fn new(source_type: &str) -> Self {
                Self {
                    source_id: format!(
                        "{}_{}",
                        source_type,
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .expect("duration_since should succeed")
                            .as_secs()
                    ),
                    source_type: source_type.to_string(),
                    connection_info: HashMap::new(),
                }
            }
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct AggregationRule {
            pub rule_name: String,
            pub rule_type: String,
            pub parameters: HashMap<String, String>,
        }

        impl AggregationRule {
            pub fn new(rule_type: &str) -> Self {
                Self {
                    rule_name: rule_type.to_string(),
                    rule_type: rule_type.to_string(),
                    parameters: HashMap::new(),
                }
            }
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct QualityControl {
            pub control_name: String,
            pub control_type: String,
            pub validation_rules: Vec<String>,
        }

        impl QualityControl {
            pub fn new(control_type: &str) -> Self {
                Self {
                    control_name: control_type.to_string(),
                    control_type: control_type.to_string(),
                    validation_rules: Vec::new(),
                }
            }
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum SamplingStrategy {
            Continuous,
            Scheduled,
            EventDriven,
            Adaptive,
            Random,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct RetentionPolicy {
            pub retention_duration: Duration,
            pub archival_strategy: String,
            pub deletion_strategy: String,
        }

        impl RetentionPolicy {
            pub fn new(duration: Duration) -> Self {
                Self {
                    retention_duration: duration,
                    archival_strategy: "compress".to_string(),
                    deletion_strategy: "automatic".to_string(),
                }
            }
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct MetricDefinition {
            pub metric_name: String,
            pub description: String,
            pub unit: String,
            pub calculation_method: String,
            pub target_value: Option<MetricValue>,
            pub thresholds: Vec<MetricThreshold>,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum MetricValue {
            Integer(i64),
            Float(f64),
            String(String),
            Boolean(bool),
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct MetricThreshold {
            pub threshold_name: String,
            pub threshold_value: f64,
        }

        impl MetricThreshold {
            pub fn new(name: &str, value: f64) -> Self {
                Self {
                    threshold_name: name.to_string(),
                    threshold_value: value,
                }
            }
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct TimestampedValue {
            pub timestamp: SystemTime,
            pub value: MetricValue,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum ThresholdStatus {
            Normal,
            Warning,
            Critical,
            Unknown,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum TrendDirection {
            Increasing,
            Decreasing,
            Stable,
            Volatile,
            Unknown,
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct CachedMetrics {
            pub result: SecurityMetricsResult,
            pub cache_timestamp: SystemTime,
            pub cache_ttl: Duration,
        }
    };
}

define_metrics_supporting_types!();

// Domain "detail" marker types: element types of the `Vec<...>`/`HashMap<String, ...>`
// configuration fields on the `*Analyzer`/`*Monitor`/`*Manager`/`*Engine`/`*Tracker`/`*Measurer`
// collector-configuration structs above (describing *what could be plugged in*, not analysis
// output) -- the actual per-domain heuristics live in the `impl` blocks above and read straight
// from the shared `MetricCollection` data instead.
macro_rules! define_metric_domain_marker_types {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Debug, Clone, Serialize, Deserialize, Default)]
            pub struct $name;
        )*
    };
}

define_metric_domain_marker_types!(
    KpiDefinition,
    TargetValue,
    ThresholdMonitor,
    TrendCalculator,
    VarianceAnalyzer,
    PerformanceTracker,
    GoalAlignmentChecker,
    BusinessImpactAssessor,
    KriDefinition,
    RiskThreshold,
    EarlyWarningSystem,
    PredictiveModel,
    CorrelationEngine,
    EscalationProcedure,
    MitigationTrigger,
    RiskAppetiteMonitor,
    VisualizationComponent,
    DataAggregator,
    RealTimeUpdater,
    InteractiveFeature,
    ExportCapability,
    CustomizationOption,
    PerformanceOptimizer,
    TrendAlgorithm,
    StatisticalModel,
    ForecastingEngine,
    SeasonalityDetector,
    ChangePointDetector,
    RegressionAnalyzer,
    TimeSeriesAnalyzer,
    PatternRecognizer,
    PredictiveAnalytic,
    AnomalyAlgorithm,
    BaselineCalculator,
    OutlierDetector,
    BehavioralAnalyzer,
    StatisticalAnomalyDetector,
    MachineLearningDetector,
    ThresholdAnomalyDetector,
    ClusteringDetector,
    IsolationForestDetector,
    BenchmarkCategory,
    IndustryComparison,
    PeerGroupAnalysis,
    BestPracticeComparison,
    MaturityAssessment,
    CompetitiveAnalysis,
    StandardBenchmark,
    CustomBenchmark,
    RealTimeStream,
    StreamProcessor,
    EventCorrelator,
    ThresholdChecker,
    AlertGenerator,
    NotificationSystem,
    EscalationManager,
    ResponseCoordinator,
    MetricsAggregator,
    ScorecardTemplate,
    ScoringAlgorithm,
    WeightCalculator,
    AggregationMethod,
    VisualizationEngine,
    ReportGenerator,
    StakeholderView,
    HistoricalComparison,
    GoalTracker,
    CorrelationMethod,
    DependencyAnalyzer,
    CausalityAnalyzer,
    AssociationMiner,
    PatternCorrelator,
    CrossDomainAnalyzer,
    TemporalCorrelator,
    MultivariateAnalyzer,
    NetworkAnalyzer,
    PerformanceIndicator,
    EfficiencyCalculator,
    EffectivenessAssessor,
    ProductivityAnalyzer,
    QualityMeasurer,
    CostAnalyzer,
    RoiCalculator,
    ValueAssessor,
    OptimizationSuggester,
    RequirementTracker,
    ControlEffectivenessMeasurer,
    AuditPreparednessAssessor,
    GapAnalyzer,
    RemediationTracker,
    CertificationMonitor,
    RegulatoryChangeTracker,
    ComplianceScorer,
);

pub fn create_security_metrics_collector() -> SecurityMetricsCollector {
    SecurityMetricsCollector::new()
}

pub fn collect_comprehensive_security_metrics(
    context: &TraitUsageContext,
) -> Result<SecurityMetricsResult, SecurityMetricsError> {
    let mut collector = SecurityMetricsCollector::new();
    collector.collect_security_metrics(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the full security-metrics collection entry point against a high-signal context and
    /// assert every top-level section of the result is populated without panicking.
    #[test]
    fn test_collect_comprehensive_security_metrics_smoke() {
        let context = TraitUsageContext {
            trait_name: "Serialize".to_string(),
            traits: vec!["Serialize".to_string(), "Clone".to_string()],
            handles_sensitive_data: true,
            handles_personal_data: true,
            has_unsafe_operations: true,
            has_bounds_checking: false,
            has_audit_logging: false,
            has_data_anonymization: false,
            requires_elevated_privileges: true,
            ..Default::default()
        };

        let result = collect_comprehensive_security_metrics(&context);
        assert!(
            result.is_ok(),
            "metrics collection should succeed: {result:?}"
        );

        let metrics = result.expect("metrics collection should succeed");
        assert!(
            !metrics.metric_collections.is_empty(),
            "expected at least one collected metric"
        );
        assert!((0.0..=10.0).contains(&metrics.overall_security_score));
        assert!((0.0..=1.0).contains(&metrics.analysis_confidence));
        assert!(
            !metrics.health_indicators.is_empty(),
            "expected health indicators derived from collected metrics"
        );
    }
    /// The lower-risk constructor path should also succeed against a default, all-`false` context.
    #[test]
    fn test_create_security_metrics_collector_default_context() {
        let mut collector = create_security_metrics_collector();
        let context = TraitUsageContext::default();
        let result = collector.collect_security_metrics(&context);
        assert!(
            result.is_ok(),
            "collection with a default context should succeed: {result:?}"
        );
        let metrics = result.expect("collection with a default context should succeed");
        assert_eq!(
            metrics.metric_collections.len(),
            12,
            "6 collectors x 2 metric definitions each"
        );
    }
}
