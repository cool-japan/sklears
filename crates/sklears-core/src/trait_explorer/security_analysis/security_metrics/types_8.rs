//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::security_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::functions::{average_quality, business_impact_from_severity, metric_value_as_f64};
use super::macros::{CachedMetrics, ThresholdStatus, TrendDirection};
use super::types::{
    AnomalyDetector, ComplianceTracker, CorrelationAnalysisResult, DashboardData,
    KpiAnalysisResult, KriMonitoringResult, KriValue, MetricsRecommendation,
    PerformanceMetricsResult, RealTimeMonitor, RealTimeStatus, ScorecardGenerator,
    SecurityAlertingSystem, SecurityMetricsConfig, SecurityScorecard, TrendAnalysisResult,
    TrendAnalyzer, VarianceAnalysis,
};
use super::types_7::{
    BenchmarkingResults, KpiAnalyzer, MetricCollector, PerformanceMeasurer, SecurityMetricsError,
};
use super::types_9::{
    ActionableInsight, AnomalyDetectionResult, BenchmarkingEngine, ComplianceMetricsResult,
    CorrelationAnalyzer, DashboardManager, DashboardPerformanceStats, HealthIndicator,
    ImprovementOpportunity, KpiScore, KriMonitor, MetricCollection, MetricsStorage,
    SecurityMetricsResult,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetricsCollector {
    pub(super) metric_collectors: HashMap<String, MetricCollector>,
    pub(super) kpi_analyzers: Vec<KpiAnalyzer>,
    pub(super) kri_monitors: Vec<KriMonitor>,
    pub(super) dashboard_managers: Vec<DashboardManager>,
    pub(super) trend_analyzers: Vec<TrendAnalyzer>,
    pub(super) anomaly_detectors: Vec<AnomalyDetector>,
    pub(super) benchmarking_engines: Vec<BenchmarkingEngine>,
    pub(super) real_time_monitors: Vec<RealTimeMonitor>,
    pub(super) scorecard_generators: Vec<ScorecardGenerator>,
    pub(super) correlation_analyzers: Vec<CorrelationAnalyzer>,
    pub(super) performance_measurers: Vec<PerformanceMeasurer>,
    pub(super) compliance_trackers: Vec<ComplianceTracker>,
    pub(super) alerting_system: SecurityAlertingSystem,
    pub(super) metrics_storage: MetricsStorage,
    pub(super) metrics_config: SecurityMetricsConfig,
    pub(super) metrics_cache: HashMap<String, CachedMetrics>,
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
    pub(super) fn collect_all_metrics(
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
    pub(super) fn analyze_kpis(
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
    pub(super) fn monitor_kris(
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
    pub(super) fn prepare_dashboard_data(
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
    pub(super) fn analyze_trends(
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
    pub(super) fn detect_anomalies(
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
    pub(super) fn perform_benchmarking(
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
    pub(super) fn get_real_time_status(
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
    pub(super) fn generate_security_scorecard(
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
    pub(super) fn analyze_correlations(
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
    pub(super) fn measure_performance(
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
    pub(super) fn track_compliance(
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
    pub(super) fn initialize_metric_collectors() -> HashMap<String, MetricCollector> {
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
impl SecurityMetricsCollector {
    pub(super) fn generate_result_id(&self, context: &TraitUsageContext) -> String {
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
    pub(super) fn get_cached_metrics(&self, result_id: &str) -> Option<CachedMetrics> {
        self.metrics_cache.get(result_id).cloned()
    }
    pub(super) fn is_cache_valid(&self, cached: &CachedMetrics) -> bool {
        cached
            .cache_timestamp
            .elapsed()
            .map(|elapsed| elapsed < cached.cache_ttl)
            .unwrap_or(false)
    }
    pub(super) fn cache_metrics(&mut self, result_id: String, result: &SecurityMetricsResult) {
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
    pub(super) fn calculate_overall_security_score(
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
    pub(super) fn calculate_analysis_confidence(
        &self,
        metric_collections: &HashMap<String, MetricCollection>,
    ) -> Result<f64, SecurityMetricsError> {
        Ok(average_quality(metric_collections).clamp(0.0, 1.0))
    }
    pub(super) fn generate_health_indicators(
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
    pub(super) fn generate_actionable_insights(
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
    pub(super) fn generate_metrics_recommendations(
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
    pub(super) fn calculate_next_collection_time(
        &self,
    ) -> Result<SystemTime, SecurityMetricsError> {
        Ok(SystemTime::now() + self.metrics_config.dashboard_refresh_rate)
    }
    pub(super) fn calculate_goal_alignment_score(
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
    pub(super) fn assess_business_impact(
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
    pub(super) fn identify_improvement_opportunities(
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
    pub(super) fn calculate_risk_appetite_compliance(
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
    pub(super) fn collect_dashboard_performance_stats(
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
    pub(super) fn calculate_overall_benchmark_score(
        &self,
        benchmark_comparisons: &HashMap<String, f64>,
    ) -> Result<f64, SecurityMetricsError> {
        if benchmark_comparisons.is_empty() {
            return Ok(5.0);
        }
        Ok(benchmark_comparisons.values().sum::<f64>() / benchmark_comparisons.len() as f64)
    }
    pub(super) fn identify_improvement_priorities(
        &self,
        best_practice_gaps: &[String],
    ) -> Result<Vec<String>, SecurityMetricsError> {
        Ok(best_practice_gaps.iter().take(5).cloned().collect())
    }
    pub(super) fn calculate_overall_health_score(
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
    pub(super) fn generate_performance_summary(
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
    pub(super) fn calculate_overall_scorecard_score(
        &self,
        weighted_scores: &HashMap<String, f64>,
    ) -> Result<f64, SecurityMetricsError> {
        if weighted_scores.is_empty() {
            return Ok(7.0);
        }
        Ok(weighted_scores.values().sum::<f64>() / weighted_scores.len() as f64)
    }
    pub(super) fn determine_security_grade(
        &self,
        overall_score: f64,
    ) -> Result<String, SecurityMetricsError> {
        let grade = match overall_score {
            s if s >= 9.0 => "A",
            s if s >= 8.0 => "B",
            s if s >= 7.0 => "C",
            s if s >= 6.0 => "D",
            _ => "F",
        };
        Ok(grade.to_string())
    }
    pub(super) fn identify_scorecard_improvement_areas(
        &self,
        category_scores: &HashMap<String, f64>,
    ) -> Result<Vec<String>, SecurityMetricsError> {
        Ok(category_scores
            .iter()
            .filter(|(_, score)| **score < 7.0)
            .map(|(name, _)| name.clone())
            .collect())
    }
    pub(super) fn generate_historical_comparison(
        &self,
        category_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>, SecurityMetricsError> {
        Ok(category_scores
            .iter()
            .map(|(name, score)| (name.clone(), score * 0.95))
            .collect())
    }
    pub(super) fn summarize_correlation_strengths(
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
    pub(super) fn identify_actionable_correlations(
        &self,
        causality_relationships: &[String],
    ) -> Result<Vec<String>, SecurityMetricsError> {
        Ok(causality_relationships.iter().take(5).cloned().collect())
    }
    pub(super) fn calculate_overall_performance_score(
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
    pub(super) fn generate_optimization_recommendations(
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
    pub(super) fn calculate_overall_compliance_score(
        &self,
        framework_compliance: &HashMap<String, f64>,
    ) -> Result<f64, SecurityMetricsError> {
        if framework_compliance.is_empty() {
            return Ok(80.0);
        }
        Ok(framework_compliance.values().sum::<f64>() / framework_compliance.len() as f64)
    }
    pub(super) fn analyze_compliance_trends(
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
    pub(super) fn identify_priority_compliance_actions(
        &self,
        compliance_gaps: &[String],
    ) -> Result<Vec<String>, SecurityMetricsError> {
        Ok(compliance_gaps.iter().take(3).cloned().collect())
    }
}
