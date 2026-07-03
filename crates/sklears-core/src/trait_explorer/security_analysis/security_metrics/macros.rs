//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use super::types_9::SecurityMetricsResult;

macro_rules! define_metrics_supporting_types {
    () => {
        #[derive(Debug, Clone, Serialize, Deserialize)] pub struct DataSource { pub
        source_id : String, pub source_type : String, pub connection_info : HashMap <
        String, String >, } impl DataSource { pub fn new(source_type : & str) -> Self {
        Self { source_id : format!("{}_{}", source_type, SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH).expect("duration_since should succeed")
        .as_secs()), source_type : source_type.to_string(), connection_info :
        HashMap::new(), } } } #[derive(Debug, Clone, Serialize, Deserialize)] pub struct
        AggregationRule { pub rule_name : String, pub rule_type : String, pub parameters
        : HashMap < String, String >, } impl AggregationRule { pub fn new(rule_type : &
        str) -> Self { Self { rule_name : rule_type.to_string(), rule_type : rule_type
        .to_string(), parameters : HashMap::new(), } } } #[derive(Debug, Clone,
        Serialize, Deserialize)] pub struct QualityControl { pub control_name : String,
        pub control_type : String, pub validation_rules : Vec < String >, } impl
        QualityControl { pub fn new(control_type : & str) -> Self { Self { control_name :
        control_type.to_string(), control_type : control_type.to_string(),
        validation_rules : Vec::new(), } } } #[derive(Debug, Clone, Serialize,
        Deserialize)] pub enum SamplingStrategy { Continuous, Scheduled, EventDriven,
        Adaptive, Random, } #[derive(Debug, Clone, Serialize, Deserialize)] pub struct
        RetentionPolicy { pub retention_duration : Duration, pub archival_strategy :
        String, pub deletion_strategy : String, } impl RetentionPolicy { pub fn
        new(duration : Duration) -> Self { Self { retention_duration : duration,
        archival_strategy : "compress".to_string(), deletion_strategy : "automatic"
        .to_string(), } } } #[derive(Debug, Clone, Serialize, Deserialize)] pub struct
        MetricDefinition { pub metric_name : String, pub description : String, pub unit :
        String, pub calculation_method : String, pub target_value : Option < MetricValue
        >, pub thresholds : Vec < MetricThreshold >, } #[derive(Debug, Clone, Serialize,
        Deserialize)] pub enum MetricValue { Integer(i64), Float(f64), String(String),
        Boolean(bool), } #[derive(Debug, Clone, Serialize, Deserialize)] pub struct
        MetricThreshold { pub threshold_name : String, pub threshold_value : f64, } impl
        MetricThreshold { pub fn new(name : & str, value : f64) -> Self { Self {
        threshold_name : name.to_string(), threshold_value : value, } } } #[derive(Debug,
        Clone, Serialize, Deserialize)] pub struct TimestampedValue { pub timestamp :
        SystemTime, pub value : MetricValue, } #[derive(Debug, Clone, Serialize,
        Deserialize)] pub enum ThresholdStatus { Normal, Warning, Critical, Unknown, }
        #[derive(Debug, Clone, Serialize, Deserialize)] pub enum TrendDirection {
        Increasing, Decreasing, Stable, Volatile, Unknown, } #[derive(Debug, Clone,
        Serialize, Deserialize)] pub struct CachedMetrics { pub result :
        SecurityMetricsResult, pub cache_timestamp : SystemTime, pub cache_ttl :
        Duration, }
    };
}
define_metrics_supporting_types!();
macro_rules! define_metric_domain_marker_types {
    ($($name:ident),* $(,)?) => {
        $(#[derive(Debug, Clone, Serialize, Deserialize, Default)] pub struct $name;)*
    };
}
define_metric_domain_marker_types!(
    KpiDefinition, TargetValue, ThresholdMonitor, TrendCalculator, VarianceAnalyzer,
    PerformanceTracker, GoalAlignmentChecker, BusinessImpactAssessor, KriDefinition,
    RiskThreshold, EarlyWarningSystem, PredictiveModel, CorrelationEngine,
    EscalationProcedure, MitigationTrigger, RiskAppetiteMonitor, VisualizationComponent,
    DataAggregator, RealTimeUpdater, InteractiveFeature, ExportCapability,
    CustomizationOption, PerformanceOptimizer, TrendAlgorithm, StatisticalModel,
    ForecastingEngine, SeasonalityDetector, ChangePointDetector, RegressionAnalyzer,
    TimeSeriesAnalyzer, PatternRecognizer, PredictiveAnalytic, AnomalyAlgorithm,
    BaselineCalculator, OutlierDetector, BehavioralAnalyzer, StatisticalAnomalyDetector,
    MachineLearningDetector, ThresholdAnomalyDetector, ClusteringDetector,
    IsolationForestDetector, BenchmarkCategory, IndustryComparison, PeerGroupAnalysis,
    BestPracticeComparison, MaturityAssessment, CompetitiveAnalysis, StandardBenchmark,
    CustomBenchmark, RealTimeStream, StreamProcessor, EventCorrelator, ThresholdChecker,
    AlertGenerator, NotificationSystem, EscalationManager, ResponseCoordinator,
    MetricsAggregator, ScorecardTemplate, ScoringAlgorithm, WeightCalculator,
    AggregationMethod, VisualizationEngine, ReportGenerator, StakeholderView,
    HistoricalComparison, GoalTracker, CorrelationMethod, DependencyAnalyzer,
    CausalityAnalyzer, AssociationMiner, PatternCorrelator, CrossDomainAnalyzer,
    TemporalCorrelator, MultivariateAnalyzer, NetworkAnalyzer, PerformanceIndicator,
    EfficiencyCalculator, EffectivenessAssessor, ProductivityAnalyzer, QualityMeasurer,
    CostAnalyzer, RoiCalculator, ValueAssessor, OptimizationSuggester,
    RequirementTracker, ControlEffectivenessMeasurer, AuditPreparednessAssessor,
    GapAnalyzer, RemediationTracker, CertificationMonitor, RegulatoryChangeTracker,
    ComplianceScorer,
);
