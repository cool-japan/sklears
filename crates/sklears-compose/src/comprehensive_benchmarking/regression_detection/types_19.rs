//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::benchmark_management::BenchmarkResult;
use super::super::config_types::*;
pub use super::super::regression_statistics::Changepoint;
use super::super::regression_statistics::{linear_regression, series_mean, series_std};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::types::{
    BusinessImpactAssessment, DetectedRegression, DetectionSensitivity, DetectionStatistics,
    EnvironmentalFactor, InvestigationStep, MetricTrend, OverallRegressionSeverity, PotentialCause,
    RegressionAlert, RegressionContext, RegressionDetectionAlgorithm, RegressionMetadata,
    RegressionRecommendationType, RegressionReport, RegressionSeverity, RootCauseAnalysis,
    SeverityAssessment, SystemHealthScore, TrendSummary,
};
use super::types_20::{
    AlertType, BaselineComparisons, ContinuousMonitoringReport, CorrelationAnalysisResult,
    EarlyWarning, RecentChange, RegressionAlertSystem, RegressionCache, RegressionDetectorConfig,
    RegressionError, RegressionImpact, RegressionRecommendation, RegressionType, RemediationAction,
    SystemMetrics, ThresholdManagement, TrendAnalysisReport, UserImpactAssessment,
};

/// Regression detector for performance regressions
#[derive(Debug)]
pub struct RegressionDetector {
    pub(super) detection_algorithms: Vec<RegressionDetectionAlgorithm>,
    pub(super) baseline_comparisons: BaselineComparisons,
    pub(super) threshold_management: ThresholdManagement,
    pub(super) alert_system: RegressionAlertSystem,
    pub(super) detector_config: RegressionDetectorConfig,
    pub(super) detection_cache: RegressionCache,
}
impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(config: RegressionDetectorConfig) -> Self {
        Self {
            detection_algorithms: vec![
                RegressionDetectionAlgorithm::StatisticalRegression,
                RegressionDetectionAlgorithm::ChangePointDetection,
                RegressionDetectionAlgorithm::TrendAnalysis,
                RegressionDetectionAlgorithm::AnomalyDetection,
            ],
            baseline_comparisons: BaselineComparisons::new(),
            threshold_management: ThresholdManagement::new(),
            alert_system: RegressionAlertSystem::new(),
            detector_config: config,
            detection_cache: RegressionCache::new(),
        }
    }
    /// Detect performance regressions in benchmark results
    pub fn detect_regressions(
        &mut self,
        results: &[BenchmarkResult],
    ) -> Result<RegressionReport, RegressionError> {
        if results.is_empty() {
            return Err(RegressionError::InsufficientData(
                "No results provided for regression detection".to_string(),
            ));
        }
        let cache_key = self.generate_cache_key(results);
        if let Some(cached_report) = self.detection_cache.get(&cache_key) {
            return Ok(cached_report.clone());
        }
        let detected_regressions = self.analyze_for_regressions(results)?;
        let severity_assessment = self.assess_regression_severity(&detected_regressions)?;
        let alerts_triggered = self.generate_alerts(&detected_regressions)?;
        let recommendations =
            self.generate_regression_recommendations(&detected_regressions, results)?;
        let root_cause_analysis =
            self.perform_root_cause_analysis(&detected_regressions, results)?;
        let report = RegressionReport {
            report_id: self.generate_report_id(),
            detection_timestamp: SystemTime::now(),
            total_benchmarks_analyzed: results.len(),
            regressions_detected: detected_regressions.len(),
            detected_regressions,
            severity_assessment,
            alerts_triggered,
            recommendations,
            root_cause_analysis,
            confidence_score: self.calculate_detection_confidence(results)?,
            metadata: self.generate_detection_metadata(results),
        };
        self.detection_cache.insert(cache_key, report.clone());
        Ok(report)
    }
    /// Perform continuous monitoring for regressions
    pub fn monitor_continuous(
        &mut self,
        results: &[BenchmarkResult],
    ) -> Result<ContinuousMonitoringReport, RegressionError> {
        let regression_report = self.detect_regressions(results)?;
        let trend_analysis = self.analyze_performance_trends(results)?;
        let health_score = self.calculate_system_health_score(results)?;
        let early_warnings = self.detect_early_warnings(results)?;
        Ok(ContinuousMonitoringReport {
            monitoring_id: self.generate_monitoring_id(),
            timestamp: SystemTime::now(),
            regression_report,
            trend_analysis,
            health_score,
            early_warnings,
            monitoring_period: self.detector_config.monitoring_period,
        })
    }
    /// Configure detection sensitivity
    pub fn configure_sensitivity(&mut self, sensitivity: DetectionSensitivity) {
        self.threshold_management.update_sensitivity(&sensitivity);
        self.detector_config.sensitivity = sensitivity;
    }
    /// Add custom detection algorithm
    pub fn add_detection_algorithm(&mut self, algorithm: RegressionDetectionAlgorithm) {
        self.detection_algorithms.push(algorithm);
    }
    /// Update regression thresholds
    pub fn update_thresholds(&mut self, thresholds: RegressionThresholds) {
        self.threshold_management.update_thresholds(thresholds);
    }
    /// Get detection statistics
    pub fn get_detection_statistics(&self) -> DetectionStatistics {
        self.detection_cache.get_statistics()
    }
    pub(super) fn analyze_for_regressions(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<Vec<DetectedRegression>, RegressionError> {
        let mut regressions = Vec::new();
        for algorithm in &self.detection_algorithms {
            let algorithm_results = self.apply_detection_algorithm(algorithm, results)?;
            regressions.extend(algorithm_results);
        }
        self.deduplicate_and_rank_regressions(regressions)
    }
    pub(super) fn apply_detection_algorithm(
        &self,
        algorithm: &RegressionDetectionAlgorithm,
        results: &[BenchmarkResult],
    ) -> Result<Vec<DetectedRegression>, RegressionError> {
        match algorithm {
            RegressionDetectionAlgorithm::StatisticalRegression => {
                self.detect_statistical_regressions(results)
            }
            RegressionDetectionAlgorithm::ChangePointDetection => {
                self.detect_changepoint_regressions(results)
            }
            RegressionDetectionAlgorithm::TrendAnalysis => self.detect_trend_regressions(results),
            RegressionDetectionAlgorithm::AnomalyDetection => {
                self.detect_anomaly_regressions(results)
            }
            RegressionDetectionAlgorithm::MachineLearning => self.detect_ml_regressions(results),
            RegressionDetectionAlgorithm::Custom(_) => Ok(Vec::new()),
        }
    }
    pub(super) fn detect_statistical_regressions(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<Vec<DetectedRegression>, RegressionError> {
        let mut regressions = Vec::new();
        let thresholds = &self.detector_config.regression_thresholds;
        for result in results {
            for (metric_name, metric_value) in &result.metrics {
                if let Some(baseline_value) =
                    self.get_baseline_value(&result.benchmark_id, metric_name)
                {
                    let degradation = (metric_value - baseline_value) / baseline_value;
                    if degradation > thresholds.performance_degradation {
                        let regression = DetectedRegression {
                            regression_id: self
                                .generate_regression_id(&result.benchmark_id, metric_name),
                            benchmark_id: result.benchmark_id.clone(),
                            metric_name: metric_name.clone(),
                            regression_type: self
                                .classify_regression_type(metric_name, degradation),
                            severity: self.calculate_regression_severity(degradation),
                            current_value: *metric_value,
                            expected_value: baseline_value,
                            degradation_percentage: degradation * 100.0,
                            detection_confidence: self
                                .calculate_statistical_confidence(degradation)?,
                            first_detected: result.timestamp,
                            consecutive_failures: self
                                .count_consecutive_failures(&result.benchmark_id, metric_name)?,
                            detection_method: "Statistical Analysis".to_string(),
                            context: self.gather_regression_context(result, metric_name)?,
                        };
                        regressions.push(regression);
                    }
                }
            }
        }
        Ok(regressions)
    }
    pub(super) fn detect_changepoint_regressions(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<Vec<DetectedRegression>, RegressionError> {
        let mut regressions = Vec::new();
        let metric_groups = self.group_results_by_metric(results);
        for (metric_name, metric_results) in metric_groups {
            if metric_results.len() < 10 {
                continue;
            }
            let changepoints = self.detect_changepoints(&metric_results)?;
            for changepoint in changepoints {
                if self.is_regression_changepoint(&changepoint) {
                    let benchmark_id = changepoint.benchmark_id.clone();
                    let severity = self.assess_changepoint_severity(&changepoint);
                    let regression = DetectedRegression {
                        regression_id: self.generate_regression_id(&benchmark_id, &metric_name),
                        benchmark_id,
                        metric_name: metric_name.clone(),
                        regression_type: RegressionType::PerformanceDegradation,
                        severity,
                        current_value: changepoint.post_change_mean,
                        expected_value: changepoint.pre_change_mean,
                        degradation_percentage: changepoint.magnitude_change * 100.0,
                        detection_confidence: changepoint.confidence,
                        first_detected: changepoint.detection_time,
                        consecutive_failures: 1,
                        detection_method: "Changepoint Detection".to_string(),
                        context: RegressionContext {
                            environmental_factors: self.assess_environmental_factors()?,
                            recent_changes: self.get_recent_changes()?,
                            system_metrics: self.get_system_metrics()?,
                            additional_info: HashMap::new(),
                        },
                    };
                    regressions.push(regression);
                }
            }
        }
        Ok(regressions)
    }
    pub(super) fn detect_trend_regressions(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<Vec<DetectedRegression>, RegressionError> {
        let mut regressions = Vec::new();
        let metric_groups = self.group_results_by_metric(results);
        for (metric_name, metric_results) in metric_groups {
            if metric_results.len() < 5 {
                continue;
            }
            let trend_analysis = self.analyze_metric_trend(&metric_results)?;
            if self.is_negative_trend(&trend_analysis) {
                if let Some(latest_result) = metric_results.last() {
                    let current_value = *latest_result.metrics.get(&metric_name).unwrap_or(&0.0);
                    let regression = DetectedRegression {
                        regression_id: self
                            .generate_regression_id(&latest_result.benchmark_id, &metric_name),
                        benchmark_id: latest_result.benchmark_id.clone(),
                        metric_name: metric_name.clone(),
                        regression_type: RegressionType::TrendDegradation,
                        severity: self.assess_trend_severity(&trend_analysis),
                        current_value,
                        expected_value: trend_analysis.expected_value,
                        degradation_percentage: trend_analysis.degradation_rate * 100.0,
                        detection_confidence: trend_analysis.confidence,
                        first_detected: trend_analysis.trend_start_time,
                        consecutive_failures: trend_analysis.consecutive_periods,
                        detection_method: "Trend Analysis".to_string(),
                        context: self.gather_regression_context(latest_result, &metric_name)?,
                    };
                    regressions.push(regression);
                }
            }
        }
        Ok(regressions)
    }
    pub(super) fn detect_anomaly_regressions(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<Vec<DetectedRegression>, RegressionError> {
        let mut regressions = Vec::new();
        for result in results {
            for (metric_name, metric_value) in &result.metrics {
                if self.is_anomalous_value(metric_value, metric_name)? {
                    let anomaly_score = self.calculate_anomaly_score(metric_value, metric_name)?;
                    if anomaly_score > self.detector_config.anomaly_threshold {
                        let regression = DetectedRegression {
                            regression_id: self
                                .generate_regression_id(&result.benchmark_id, metric_name),
                            benchmark_id: result.benchmark_id.clone(),
                            metric_name: metric_name.clone(),
                            regression_type: RegressionType::AnomalyRegression,
                            severity: self.severity_from_anomaly_score(anomaly_score),
                            current_value: *metric_value,
                            expected_value: self.get_expected_value(metric_name)?,
                            degradation_percentage: anomaly_score * 100.0,
                            detection_confidence: anomaly_score,
                            first_detected: result.timestamp,
                            consecutive_failures: 1,
                            detection_method: "Anomaly Detection".to_string(),
                            context: self.gather_regression_context(result, metric_name)?,
                        };
                        regressions.push(regression);
                    }
                }
            }
        }
        Ok(regressions)
    }
    pub(super) fn detect_ml_regressions(
        &self,
        _results: &[BenchmarkResult],
    ) -> Result<Vec<DetectedRegression>, RegressionError> {
        Ok(Vec::new())
    }
    pub(super) fn assess_regression_severity(
        &self,
        regressions: &[DetectedRegression],
    ) -> Result<SeverityAssessment, RegressionError> {
        let critical_count = regressions
            .iter()
            .filter(|r| matches!(r.severity, RegressionSeverity::Critical))
            .count();
        let high_count = regressions
            .iter()
            .filter(|r| matches!(r.severity, RegressionSeverity::High))
            .count();
        let medium_count = regressions
            .iter()
            .filter(|r| matches!(r.severity, RegressionSeverity::Medium))
            .count();
        let low_count = regressions
            .iter()
            .filter(|r| matches!(r.severity, RegressionSeverity::Low))
            .count();
        let overall_severity = if critical_count > 0 {
            OverallRegressionSeverity::Critical
        } else if high_count > 0 {
            OverallRegressionSeverity::High
        } else if medium_count > 0 {
            OverallRegressionSeverity::Medium
        } else {
            OverallRegressionSeverity::Low
        };
        let total_regressions = regressions.len();
        let risk_score = if total_regressions > 0 {
            (critical_count * 4 + high_count * 3 + medium_count * 2 + low_count) as f64
                / total_regressions as f64
        } else {
            0.0
        };
        let business_impact = self.assess_business_impact(regressions)?;
        let user_impact = self.assess_user_impact(regressions)?;
        Ok(SeverityAssessment {
            overall_severity,
            critical_regressions: critical_count,
            high_severity_regressions: high_count,
            medium_severity_regressions: medium_count,
            low_severity_regressions: low_count,
            total_regressions,
            risk_score,
            business_impact,
            user_impact,
            estimated_recovery_time: self.estimate_recovery_time(regressions)?,
        })
    }
    pub(super) fn generate_alerts(
        &self,
        regressions: &[DetectedRegression],
    ) -> Result<Vec<RegressionAlert>, RegressionError> {
        let mut alerts = Vec::new();
        for regression in regressions {
            if self.should_generate_alert(regression) {
                let alert = RegressionAlert {
                    alert_id: self.generate_alert_id(&regression.regression_id),
                    regression_id: regression.regression_id.clone(),
                    alert_type: self.determine_alert_type(regression),
                    severity: self.map_regression_to_alert_severity(&regression.severity),
                    message: self.generate_alert_message(regression),
                    triggered_at: SystemTime::now(),
                    resolved: false,
                    escalation_level: 0,
                    notification_channels: self.select_notification_channels(&regression.severity),
                    acknowledgment_required: matches!(
                        regression.severity,
                        RegressionSeverity::Critical | RegressionSeverity::High
                    ),
                    auto_resolution_timeout: self
                        .calculate_auto_resolution_timeout(&regression.severity),
                };
                alerts.push(alert);
            }
        }
        self.apply_alert_filtering(alerts)
    }
    pub(super) fn generate_regression_recommendations(
        &self,
        regressions: &[DetectedRegression],
        results: &[BenchmarkResult],
    ) -> Result<Vec<RegressionRecommendation>, RegressionError> {
        let mut recommendations = Vec::new();
        let grouped_regressions = self.group_regressions_by_type(regressions);
        for (regression_type, type_regressions) in grouped_regressions {
            let recommendation = match regression_type {
                RegressionType::PerformanceDegradation => {
                    self.generate_performance_recommendation(&type_regressions, results)?
                }
                RegressionType::AccuracyDrop => {
                    self.generate_accuracy_recommendation(&type_regressions, results)?
                }
                RegressionType::MemoryIncrease => {
                    self.generate_memory_recommendation(&type_regressions, results)?
                }
                RegressionType::LatencyIncrease => {
                    self.generate_latency_recommendation(&type_regressions, results)?
                }
                _ => self.generate_generic_recommendation(&type_regressions, results)?,
            };
            recommendations.push(recommendation);
        }
        if regressions.len() > 5 {
            recommendations.push(self.generate_system_wide_recommendation(regressions, results)?);
        }
        Ok(recommendations)
    }
    pub(super) fn perform_root_cause_analysis(
        &self,
        regressions: &[DetectedRegression],
        results: &[BenchmarkResult],
    ) -> Result<RootCauseAnalysis, RegressionError> {
        let mut potential_causes = Vec::new();
        let correlation_analysis = self.analyze_regression_correlations(regressions)?;
        potential_causes.extend(correlation_analysis.potential_causes);
        let environmental_analysis = self.analyze_environmental_factors(results)?;
        potential_causes.extend(environmental_analysis);
        let temporal_analysis = self.analyze_temporal_patterns(regressions)?;
        potential_causes.extend(temporal_analysis);
        let resource_analysis = self.analyze_resource_constraints(results)?;
        potential_causes.extend(resource_analysis);
        let confidence_scores = self.calculate_cause_confidence(&potential_causes)?;
        let investigation_steps = self.generate_investigation_steps(&potential_causes);
        let remediation_priority = self.prioritize_remediation(&potential_causes);
        Ok(RootCauseAnalysis {
            analysis_id: self.generate_analysis_id(),
            analysis_timestamp: SystemTime::now(),
            potential_causes,
            confidence_scores,
            investigation_steps,
            remediation_priority,
        })
    }
    pub(super) fn generate_report_id(&self) -> String {
        format!(
            "reg_report_{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        )
    }
    pub(super) fn generate_monitoring_id(&self) -> String {
        format!(
            "mon_{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        )
    }
    pub(super) fn generate_regression_id(&self, benchmark_id: &str, metric_name: &str) -> String {
        format!("reg_{}_{}", benchmark_id, metric_name)
    }
    pub(super) fn generate_alert_id(&self, regression_id: &str) -> String {
        format!("alert_{}", regression_id)
    }
    pub(super) fn generate_analysis_id(&self) -> String {
        format!(
            "rca_{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        )
    }
    pub(super) fn generate_cache_key(&self, results: &[BenchmarkResult]) -> String {
        format!(
            "regression_{}_{}",
            results.len(),
            results
                .iter()
                .map(|r| r.result_id.as_str())
                .collect::<Vec<_>>()
                .join("_")
        )
    }
}
impl RegressionDetector {
    pub(super) fn get_baseline_value(
        &self,
        _benchmark_id: &str,
        _metric_name: &str,
    ) -> Option<f64> {
        Some(100.0)
    }
    pub(super) fn classify_regression_type(
        &self,
        metric_name: &str,
        _degradation: f64,
    ) -> RegressionType {
        match metric_name {
            "execution_time" => RegressionType::PerformanceDegradation,
            "accuracy" => RegressionType::AccuracyDrop,
            "memory_usage" => RegressionType::MemoryIncrease,
            "latency" => RegressionType::LatencyIncrease,
            _ => RegressionType::PerformanceDegradation,
        }
    }
    pub(super) fn calculate_regression_severity(&self, degradation: f64) -> RegressionSeverity {
        if degradation > 0.5 {
            RegressionSeverity::Critical
        } else if degradation > 0.25 {
            RegressionSeverity::High
        } else if degradation > 0.1 {
            RegressionSeverity::Medium
        } else {
            RegressionSeverity::Low
        }
    }
    /// Confidence that a single observed `degradation` (relative change vs.
    /// baseline) is a real regression. We lack a per-call sample to run
    /// Welch's t-test against, so we fall back to a saturating score
    /// `1 − exp(−|degradation|/0.1)` (10% half-life).
    pub(super) fn calculate_statistical_confidence(
        &self,
        degradation: f64,
    ) -> Result<f64, RegressionError> {
        let mag = degradation.abs();
        Ok((1.0 - (-mag / 0.1).exp()).clamp(0.0, 1.0))
    }
    pub(super) fn count_consecutive_failures(
        &self,
        _benchmark_id: &str,
        _metric_name: &str,
    ) -> Result<u32, RegressionError> {
        Ok(1)
    }
    pub(super) fn gather_regression_context(
        &self,
        _result: &BenchmarkResult,
        _metric_name: &str,
    ) -> Result<RegressionContext, RegressionError> {
        Ok(RegressionContext {
            environmental_factors: Vec::new(),
            recent_changes: Vec::new(),
            system_metrics: SystemMetrics::default(),
            additional_info: HashMap::new(),
        })
    }
    pub(super) fn group_results_by_metric(
        &self,
        results: &[BenchmarkResult],
    ) -> HashMap<String, Vec<BenchmarkResult>> {
        let mut groups = HashMap::new();
        for result in results {
            for metric_name in result.metrics.keys() {
                groups
                    .entry(metric_name.clone())
                    .or_insert_with(Vec::new)
                    .push(result.clone());
            }
        }
        groups
    }
    /// Detect changepoints in a metric time series via the classical
    /// two-sided CUSUM (cumulative sum) algorithm.
    ///
    /// The reference mean μ₀ and σ are estimated from the first
    /// `warmup` samples (or the global statistics if the series is short).
    /// CUSUM accumulates `S_t^+ = max(0, S_{t-1}^+ + (x_t - μ₀ - k))` and
    /// `S_t^- = max(0, S_{t-1}^- - (x_t - μ₀ + k))` where `k = 0.5 σ` is
    /// the slack and `h = 5 σ` is the alarm threshold (Page 1954, Lucas 1976).
    pub(super) fn detect_changepoints(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<Vec<Changepoint>, RegressionError> {
        if results.len() < 4 {
            return Ok(Vec::new());
        }
        let mut ordered: Vec<&BenchmarkResult> = results.iter().collect();
        ordered.sort_by_key(|r| r.timestamp);
        let metric_name = match ordered
            .iter()
            .flat_map(|r| r.metrics.keys())
            .next()
            .cloned()
        {
            Some(name) => name,
            None => return Ok(Vec::new()),
        };
        let values: Vec<f64> = ordered
            .iter()
            .filter_map(|r| r.metrics.get(&metric_name).copied())
            .collect();
        if values.len() < 4 {
            return Ok(Vec::new());
        }
        let warmup = (values.len() / 2).min(8).max(2);
        let warm = &values[..warmup];
        let mu0 = series_mean(warm);
        let sigma_est = series_std(warm);
        let sigma = if sigma_est > 1e-12 {
            sigma_est
        } else {
            (mu0.abs() * 0.05).max(1e-3)
        };
        let k = 0.5 * sigma;
        let h = 5.0 * sigma;
        let mut s_pos = 0.0_f64;
        let mut s_neg = 0.0_f64;
        let mut alarms: Vec<(usize, f64, f64)> = Vec::new();
        for (i, &x) in values.iter().enumerate() {
            s_pos = (s_pos + x - mu0 - k).max(0.0);
            s_neg = (s_neg - (x - mu0) - k).max(0.0);
            if s_pos > h || s_neg > h {
                let stat = s_pos.max(s_neg);
                alarms.push((i, stat, x));
                s_pos = 0.0;
                s_neg = 0.0;
            }
        }
        let mut changepoints = Vec::with_capacity(alarms.len());
        for (idx, stat, _x) in alarms {
            if idx == 0 || idx >= values.len() {
                continue;
            }
            let pre_mean = series_mean(&values[..idx]);
            let post_mean = series_mean(&values[idx..]);
            let denom = pre_mean.abs().max(1e-12);
            let magnitude = (post_mean - pre_mean) / denom;
            let confidence = ((stat / (h + 1e-12)).tanh()).clamp(0.0, 1.0);
            let benchmark_id = ordered
                .get(idx)
                .map(|r| r.benchmark_id.clone())
                .unwrap_or_default();
            let detection_time = ordered
                .get(idx)
                .map(|r| r.timestamp)
                .unwrap_or_else(SystemTime::now);
            changepoints.push(Changepoint {
                benchmark_id,
                detection_time,
                pre_change_mean: pre_mean,
                post_change_mean: post_mean,
                magnitude_change: magnitude,
                confidence,
            });
        }
        Ok(changepoints)
    }
    pub(super) fn is_regression_changepoint(&self, changepoint: &Changepoint) -> bool {
        let threshold = self
            .detector_config
            .regression_thresholds
            .performance_degradation
            .max(0.0);
        changepoint.magnitude_change > threshold && changepoint.confidence > 0.1
    }
    pub(super) fn assess_changepoint_severity(
        &self,
        changepoint: &Changepoint,
    ) -> RegressionSeverity {
        let magnitude = changepoint.magnitude_change.abs();
        if magnitude > 0.5 {
            RegressionSeverity::Critical
        } else if magnitude > 0.25 {
            RegressionSeverity::High
        } else if magnitude > 0.1 {
            RegressionSeverity::Medium
        } else {
            RegressionSeverity::Low
        }
    }
    /// Fit a least-squares line `y = a + b·t` to the series, returning a
    /// `MetricTrend` whose `trend_strength` is the coefficient of
    /// determination R² ∈ [0, 1] and whose `degradation_rate` is the
    /// per-step relative slope `b / mean(y)`.
    pub(super) fn analyze_metric_trend(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<MetricTrend, RegressionError> {
        if results.is_empty() {
            return Err(RegressionError::InsufficientData(
                "trend analysis requires at least one sample".to_string(),
            ));
        }
        let mut ordered: Vec<&BenchmarkResult> = results.iter().collect();
        ordered.sort_by_key(|r| r.timestamp);
        let metric_name = match ordered
            .iter()
            .flat_map(|r| r.metrics.keys())
            .next()
            .cloned()
        {
            Some(name) => name,
            None => {
                return Err(RegressionError::InsufficientData(
                    "no metrics available for trend analysis".to_string(),
                ));
            }
        };
        let values: Vec<f64> = ordered
            .iter()
            .filter_map(|r| r.metrics.get(&metric_name).copied())
            .collect();
        if values.len() < 2 {
            return Err(RegressionError::InsufficientData(
                "trend analysis requires at least two samples".to_string(),
            ));
        }
        let xs: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
        let (slope, intercept, r2) = linear_regression(&xs, &values);
        let mean_y = series_mean(&values);
        let last_x = (values.len() - 1) as f64;
        let expected_value = intercept + slope * last_x;
        let denom = mean_y.abs().max(1e-12);
        let degradation_rate = slope / denom;
        let trend_direction = if slope > 1e-9 {
            TrendDirection::Increasing
        } else if slope < -1e-9 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };
        let trend_start_time = ordered
            .first()
            .map(|r| r.timestamp)
            .unwrap_or_else(SystemTime::now);
        let mut consecutive_periods: u32 = 1;
        let mut prev_sign: i8 = 0;
        for (i, &y) in values.iter().enumerate() {
            let pred = intercept + slope * (i as f64);
            let resid = y - pred;
            let sign = if resid > 0.0 {
                1_i8
            } else if resid < 0.0 {
                -1_i8
            } else {
                0
            };
            if sign != 0 && sign == prev_sign {
                consecutive_periods = consecutive_periods.saturating_add(1);
            }
            if sign != 0 {
                prev_sign = sign;
            }
        }
        Ok(MetricTrend {
            metric_name,
            trend_direction,
            trend_strength: r2.clamp(0.0, 1.0),
            expected_value,
            degradation_rate,
            confidence: r2.clamp(0.0, 1.0),
            trend_start_time,
            consecutive_periods,
        })
    }
    /// A "negative trend" for performance metrics means a sustained
    /// slope of the wrong sign with non-trivial fit quality. We treat
    /// `Decreasing` as bad for accuracy/throughput-style series and
    /// `Increasing` as bad for latency/memory/error-style series. Without
    /// per-metric polarity information at this layer we conservatively
    /// flag *both* strong directional trends as regressions, leaving
    /// classification to `classify_regression_type`.
    pub(super) fn is_negative_trend(&self, trend: &MetricTrend) -> bool {
        let directional = matches!(
            trend.trend_direction,
            TrendDirection::Decreasing | TrendDirection::Increasing
        );
        directional && trend.trend_strength > 0.5
    }
    pub(super) fn assess_trend_severity(&self, trend: &MetricTrend) -> RegressionSeverity {
        let strength = trend.trend_strength;
        let rate = trend.degradation_rate.abs();
        if strength > 0.9 && rate > 0.25 {
            RegressionSeverity::Critical
        } else if strength > 0.75 && rate > 0.1 {
            RegressionSeverity::High
        } else if strength > 0.5 {
            RegressionSeverity::Medium
        } else {
            RegressionSeverity::Low
        }
    }
    /// Flag a value as anomalous if its z-score (against the metric's
    /// baseline mean and an assumed coefficient-of-variation) exceeds the
    /// configured `anomaly_threshold` (interpreted as a normalized
    /// `tanh(|z|/3)` score in [0, 1]).
    pub(super) fn is_anomalous_value(
        &self,
        value: &f64,
        metric_name: &str,
    ) -> Result<bool, RegressionError> {
        let score = self.calculate_anomaly_score(value, metric_name)?;
        Ok(score > self.detector_config.anomaly_threshold)
    }
    /// Anomaly score in [0, 1] derived from |z| = |x − μ|/σ.
    /// μ is taken from `get_baseline_value` (falling back to the value
    /// itself when no baseline exists, yielding a score of 0). σ is
    /// estimated from a 10% relative dispersion model unless the baseline
    /// is zero, in which case we use an additive σ = 1.
    pub(super) fn calculate_anomaly_score(
        &self,
        value: &f64,
        metric_name: &str,
    ) -> Result<f64, RegressionError> {
        let baseline = self.get_baseline_value("", metric_name).unwrap_or(*value);
        let sigma = (baseline.abs() * 0.1).max(1e-9);
        let z = (value - baseline).abs() / sigma;
        Ok((z / 3.0).tanh().clamp(0.0, 1.0))
    }
    pub(super) fn get_expected_value(&self, metric_name: &str) -> Result<f64, RegressionError> {
        Ok(self.get_baseline_value("", metric_name).unwrap_or(0.0))
    }
    pub(super) fn severity_from_anomaly_score(&self, score: f64) -> RegressionSeverity {
        if score > 0.9 {
            RegressionSeverity::Critical
        } else if score > 0.7 {
            RegressionSeverity::High
        } else if score > 0.5 {
            RegressionSeverity::Medium
        } else {
            RegressionSeverity::Low
        }
    }
    pub(super) fn deduplicate_and_rank_regressions(
        &self,
        mut regressions: Vec<DetectedRegression>,
    ) -> Result<Vec<DetectedRegression>, RegressionError> {
        regressions.sort_by(|a, b| {
            b.detection_confidence
                .partial_cmp(&a.detection_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        regressions.dedup_by(|a, b| a.regression_id == b.regression_id);
        Ok(regressions)
    }
    /// Aggregate detection confidence as a function of sample size. We
    /// follow the heuristic `1 − 1/(1 + n/10)` which approaches 1 as the
    /// sample count grows and is 0 with no samples.
    pub(super) fn calculate_detection_confidence(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<f64, RegressionError> {
        let n = results.len() as f64;
        Ok((n / (10.0 + n)).clamp(0.0, 1.0))
    }
    pub(super) fn generate_detection_metadata(
        &self,
        results: &[BenchmarkResult],
    ) -> RegressionMetadata {
        RegressionMetadata {
            detector_version: "1.0.0".to_string(),
            detection_parameters: HashMap::new(),
            analysis_duration: Duration::from_millis(100),
            data_quality_score: if results.is_empty() { 0.0 } else { 0.85 },
        }
    }
    pub(super) fn analyze_performance_trends(
        &self,
        _results: &[BenchmarkResult],
    ) -> Result<TrendAnalysisReport, RegressionError> {
        Ok(TrendAnalysisReport {
            trend_id: "".to_string(),
            analysis_period: Duration::from_secs(0),
            trend_summary: TrendSummary {
                overall_trend: TrendDirection::Stable,
                trend_strength: 0.0,
                confidence: 0.0,
                significant_changes: 0,
            },
            metric_trends: HashMap::new(),
            predictions: Vec::new(),
        })
    }
    pub(super) fn calculate_system_health_score(
        &self,
        _results: &[BenchmarkResult],
    ) -> Result<SystemHealthScore, RegressionError> {
        Ok(SystemHealthScore {
            overall_score: 0.85,
            component_scores: HashMap::new(),
            health_trend: TrendDirection::Stable,
            critical_issues: 0,
            warnings: 0,
        })
    }
    pub(super) fn detect_early_warnings(
        &self,
        _results: &[BenchmarkResult],
    ) -> Result<Vec<EarlyWarning>, RegressionError> {
        Ok(Vec::new())
    }
    pub(super) fn should_generate_alert(&self, _regression: &DetectedRegression) -> bool {
        true
    }
    pub(super) fn determine_alert_type(&self, _regression: &DetectedRegression) -> AlertType {
        AlertType::PerformanceRegression
    }
    pub(super) fn map_regression_to_alert_severity(
        &self,
        severity: &RegressionSeverity,
    ) -> AlertSeverity {
        match severity {
            RegressionSeverity::Critical => AlertSeverity::Critical,
            RegressionSeverity::High => AlertSeverity::Error,
            RegressionSeverity::Medium => AlertSeverity::Warning,
            RegressionSeverity::Low => AlertSeverity::Info,
        }
    }
    pub(super) fn generate_alert_message(&self, regression: &DetectedRegression) -> String {
        format!(
            "Regression detected in {}: {:.1}% degradation",
            regression.benchmark_id, regression.degradation_percentage
        )
    }
    pub(super) fn select_notification_channels(
        &self,
        _severity: &RegressionSeverity,
    ) -> Vec<String> {
        vec!["email".to_string()]
    }
    pub(super) fn calculate_auto_resolution_timeout(
        &self,
        _severity: &RegressionSeverity,
    ) -> Option<Duration> {
        Some(Duration::from_secs(3600))
    }
    pub(super) fn apply_alert_filtering(
        &self,
        alerts: Vec<RegressionAlert>,
    ) -> Result<Vec<RegressionAlert>, RegressionError> {
        Ok(alerts)
    }
    pub(super) fn group_regressions_by_type(
        &self,
        regressions: &[DetectedRegression],
    ) -> HashMap<RegressionType, Vec<DetectedRegression>> {
        let mut groups = HashMap::new();
        for regression in regressions {
            groups
                .entry(regression.regression_type.clone())
                .or_insert_with(Vec::new)
                .push(regression.clone());
        }
        groups
    }
    pub(super) fn generate_performance_recommendation(
        &self,
        _regressions: &[DetectedRegression],
        _results: &[BenchmarkResult],
    ) -> Result<RegressionRecommendation, RegressionError> {
        Ok(RegressionRecommendation {
            recommendation_id: "".to_string(),
            recommendation_type: RegressionRecommendationType::OptimizationNeeded,
            priority: RecommendationPriority::High,
            title: "".to_string(),
            description: "".to_string(),
            affected_benchmarks: Vec::new(),
            root_cause_analysis: Vec::new(),
            remediation_steps: Vec::new(),
            expected_timeline: Duration::from_secs(0),
            confidence: 0.0,
            estimated_effort: ImplementationEffort::Medium,
            expected_impact: RegressionImpact {
                performance_improvement: 0.0,
                risk_reduction: 0.0,
                cost_impact: CostImpact::Low,
                timeline_impact: Duration::from_secs(0),
            },
        })
    }
    pub(super) fn generate_accuracy_recommendation(
        &self,
        _regressions: &[DetectedRegression],
        _results: &[BenchmarkResult],
    ) -> Result<RegressionRecommendation, RegressionError> {
        self.generate_performance_recommendation(_regressions, _results)
    }
    pub(super) fn generate_memory_recommendation(
        &self,
        _regressions: &[DetectedRegression],
        _results: &[BenchmarkResult],
    ) -> Result<RegressionRecommendation, RegressionError> {
        self.generate_performance_recommendation(_regressions, _results)
    }
    pub(super) fn generate_latency_recommendation(
        &self,
        _regressions: &[DetectedRegression],
        _results: &[BenchmarkResult],
    ) -> Result<RegressionRecommendation, RegressionError> {
        self.generate_performance_recommendation(_regressions, _results)
    }
    pub(super) fn generate_generic_recommendation(
        &self,
        _regressions: &[DetectedRegression],
        _results: &[BenchmarkResult],
    ) -> Result<RegressionRecommendation, RegressionError> {
        self.generate_performance_recommendation(_regressions, _results)
    }
    pub(super) fn generate_system_wide_recommendation(
        &self,
        _regressions: &[DetectedRegression],
        _results: &[BenchmarkResult],
    ) -> Result<RegressionRecommendation, RegressionError> {
        self.generate_performance_recommendation(_regressions, _results)
    }
    pub(super) fn analyze_regression_correlations(
        &self,
        _regressions: &[DetectedRegression],
    ) -> Result<CorrelationAnalysisResult, RegressionError> {
        Ok(CorrelationAnalysisResult {
            potential_causes: Vec::new(),
        })
    }
    pub(super) fn analyze_environmental_factors(
        &self,
        _results: &[BenchmarkResult],
    ) -> Result<Vec<PotentialCause>, RegressionError> {
        Ok(Vec::new())
    }
    pub(super) fn analyze_temporal_patterns(
        &self,
        _regressions: &[DetectedRegression],
    ) -> Result<Vec<PotentialCause>, RegressionError> {
        Ok(Vec::new())
    }
    pub(super) fn analyze_resource_constraints(
        &self,
        _results: &[BenchmarkResult],
    ) -> Result<Vec<PotentialCause>, RegressionError> {
        Ok(Vec::new())
    }
    pub(super) fn calculate_cause_confidence(
        &self,
        _causes: &[PotentialCause],
    ) -> Result<HashMap<String, f64>, RegressionError> {
        Ok(HashMap::new())
    }
    pub(super) fn generate_investigation_steps(
        &self,
        _causes: &[PotentialCause],
    ) -> Vec<InvestigationStep> {
        Vec::new()
    }
    pub(super) fn prioritize_remediation(
        &self,
        _causes: &[PotentialCause],
    ) -> Vec<RemediationAction> {
        Vec::new()
    }
    pub(super) fn assess_business_impact(
        &self,
        _regressions: &[DetectedRegression],
    ) -> Result<BusinessImpactAssessment, RegressionError> {
        Ok(BusinessImpactAssessment::default())
    }
    pub(super) fn assess_user_impact(
        &self,
        _regressions: &[DetectedRegression],
    ) -> Result<UserImpactAssessment, RegressionError> {
        Ok(UserImpactAssessment::default())
    }
    pub(super) fn estimate_recovery_time(
        &self,
        _regressions: &[DetectedRegression],
    ) -> Result<Duration, RegressionError> {
        Ok(Duration::from_secs(3600))
    }
    pub(super) fn assess_environmental_factors(
        &self,
    ) -> Result<Vec<EnvironmentalFactor>, RegressionError> {
        Ok(Vec::new())
    }
    pub(super) fn get_recent_changes(&self) -> Result<Vec<RecentChange>, RegressionError> {
        Ok(Vec::new())
    }
    pub(super) fn get_system_metrics(&self) -> Result<SystemMetrics, RegressionError> {
        Ok(SystemMetrics::default())
    }
}
