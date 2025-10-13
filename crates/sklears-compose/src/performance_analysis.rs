//! Performance Analysis and Advanced Analytics
//!
//! This module provides comprehensive performance analysis capabilities including
//! anomaly detection, trend analysis, performance insights generation, optimization
//! recommendations, and advanced analytics for monitoring data.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;

use crate::monitoring_core::{TimeRange, TrendDirection, PerformanceSummary, PerformanceTrends};
use crate::metrics_collection::PerformanceMetric;
use crate::event_tracking::TaskExecutionEvent;
use crate::configuration_management::PerformanceAnalysisConfig;

/// Performance analyzer for comprehensive performance analysis
///
/// Provides advanced analytics including anomaly detection, trend analysis,
/// pattern recognition, and optimization recommendations for monitoring data.
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Analysis configuration
    config: PerformanceAnalysisConfig,

    /// Anomaly detectors
    anomaly_detectors: HashMap<String, Box<dyn AnomalyDetector>>,

    /// Trend analyzers
    trend_analyzers: HashMap<String, Box<dyn TrendAnalyzer>>,

    /// Pattern recognizers
    pattern_recognizers: Vec<Box<dyn PatternRecognizer>>,

    /// Correlation analyzer
    correlation_analyzer: CorrelationAnalyzer,

    /// Performance insights cache
    insights_cache: Arc<Mutex<InsightsCache>>,

    /// Analysis statistics
    stats: AnalysisStatistics,

    /// Thread safety lock
    lock: Arc<RwLock<()>>,
}

impl PerformanceAnalyzer {
    /// Create new performance analyzer
    pub fn new(config: PerformanceAnalysisConfig) -> Self {
        Self {
            config: config.clone(),
            anomaly_detectors: HashMap::new(),
            trend_analyzers: HashMap::new(),
            pattern_recognizers: Vec::new(),
            correlation_analyzer: CorrelationAnalyzer::new(config.correlation),
            insights_cache: Arc::new(Mutex::new(InsightsCache::new())),
            stats: AnalysisStatistics::new(),
            lock: Arc::new(RwLock::new(())),
        }
    }

    /// Register anomaly detector
    pub fn register_anomaly_detector(&mut self, name: String, detector: Box<dyn AnomalyDetector>) {
        let _lock = self.lock.write().unwrap();
        self.anomaly_detectors.insert(name, detector);
    }

    /// Register trend analyzer
    pub fn register_trend_analyzer(&mut self, name: String, analyzer: Box<dyn TrendAnalyzer>) {
        let _lock = self.lock.write().unwrap();
        self.trend_analyzers.insert(name, analyzer);
    }

    /// Register pattern recognizer
    pub fn register_pattern_recognizer(&mut self, recognizer: Box<dyn PatternRecognizer>) {
        let _lock = self.lock.write().unwrap();
        self.pattern_recognizers.push(recognizer);
    }

    /// Perform comprehensive performance analysis
    pub fn analyze_performance(&mut self, metrics: &[PerformanceMetric], events: &[TaskExecutionEvent]) -> SklResult<PerformanceInsights> {
        let _lock = self.lock.write().unwrap();
        let start_time = Instant::now();

        // Check cache first
        let cache_key = self.generate_cache_key(metrics, events);
        if let Some(cached_insights) = self.get_cached_insights(&cache_key) {
            return Ok(cached_insights);
        }

        let mut insights = Vec::new();
        let mut patterns = Vec::new();
        let mut anomalies = Vec::new();
        let mut correlations = Vec::new();

        // Detect anomalies
        for (name, detector) in &self.anomaly_detectors {
            match detector.detect_anomalies(metrics) {
                Ok(detected) => {
                    anomalies.extend(detected);
                    self.stats.anomaly_detections += 1;
                }
                Err(e) => {
                    log::warn!("Anomaly detection failed for '{}': {}", name, e);
                    self.stats.analysis_failures += 1;
                }
            }
        }

        // Analyze trends
        for (name, analyzer) in &self.trend_analyzers {
            match analyzer.analyze_trends(metrics) {
                Ok(trends) => {
                    insights.extend(self.generate_trend_insights(&trends));
                    self.stats.trend_analyses += 1;
                }
                Err(e) => {
                    log::warn!("Trend analysis failed for '{}': {}", name, e);
                    self.stats.analysis_failures += 1;
                }
            }
        }

        // Recognize patterns
        for recognizer in &self.pattern_recognizers {
            match recognizer.recognize_patterns(metrics, events) {
                Ok(recognized) => {
                    patterns.extend(recognized);
                    self.stats.pattern_recognitions += 1;
                }
                Err(e) => {
                    log::warn!("Pattern recognition failed: {}", e);
                    self.stats.analysis_failures += 1;
                }
            }
        }

        // Analyze correlations
        match self.correlation_analyzer.analyze_correlations(metrics) {
            Ok(detected) => {
                correlations.extend(detected);
                self.stats.correlation_analyses += 1;
            }
            Err(e) => {
                log::warn!("Correlation analysis failed: {}", e);
                self.stats.analysis_failures += 1;
            }
        }

        // Generate performance insights
        insights.extend(self.generate_performance_insights(metrics, events, &anomalies, &patterns)?);

        let performance_insights = PerformanceInsights {
            insights,
            patterns,
            anomalies,
            correlations,
            generated_at: SystemTime::now(),
            confidence_score: self.calculate_confidence_score(&anomalies, &patterns, &correlations),
            recommendations: self.generate_recommendations(metrics, events, &anomalies, &patterns)?,
        };

        // Cache results
        self.cache_insights(cache_key, &performance_insights);

        // Update statistics
        let analysis_time = start_time.elapsed();
        self.stats.total_analyses += 1;
        self.stats.total_analysis_time += analysis_time;
        self.stats.last_analysis = SystemTime::now();

        Ok(performance_insights)
    }

    /// Generate performance insights from analysis results
    fn generate_performance_insights(&self, metrics: &[PerformanceMetric], events: &[TaskExecutionEvent], anomalies: &[Anomaly], patterns: &[Pattern]) -> SklResult<Vec<Insight>> {
        let mut insights = Vec::new();

        // Generate insights from metrics
        insights.extend(self.analyze_metric_insights(metrics)?);

        // Generate insights from events
        insights.extend(self.analyze_event_insights(events)?);

        // Generate insights from anomalies
        for anomaly in anomalies {
            insights.push(Insight {
                category: "Anomaly".to_string(),
                description: format!("Anomaly detected in {}: {}", anomaly.metric_name, anomaly.description),
                confidence: anomaly.confidence_score,
                supporting_data: vec![format!("Score: {}", anomaly.anomaly_score)],
                impact_score: anomaly.impact_score,
                urgency: if anomaly.severity == AnomalySeverity::Critical { InsightUrgency::High } else { InsightUrgency::Medium },
            });
        }

        // Generate insights from patterns
        for pattern in patterns {
            insights.push(Insight {
                category: "Pattern".to_string(),
                description: format!("Pattern detected: {}", pattern.description),
                confidence: pattern.confidence,
                supporting_data: vec![format!("Frequency: {}", pattern.frequency)],
                impact_score: pattern.impact,
                urgency: InsightUrgency::Medium,
            });
        }

        Ok(insights)
    }

    /// Analyze metrics for insights
    fn analyze_metric_insights(&self, metrics: &[PerformanceMetric]) -> SklResult<Vec<Insight>> {
        let mut insights = Vec::new();

        // Group metrics by name
        let mut metric_groups: HashMap<String, Vec<&PerformanceMetric>> = HashMap::new();
        for metric in metrics {
            metric_groups.entry(metric.name.clone()).or_insert_with(Vec::new).push(metric);
        }

        // Analyze each metric group
        for (metric_name, metric_list) in metric_groups {
            if metric_list.len() < 2 {
                continue;
            }

            let values: Vec<f64> = metric_list.iter().map(|m| m.value).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            // High variability insight
            if std_dev / mean > 0.3 && mean > 0.0 {
                insights.push(Insight {
                    category: "Variability".to_string(),
                    description: format!("High variability detected in {}", metric_name),
                    confidence: 0.8,
                    supporting_data: vec![
                        format!("Standard deviation: {:.2}", std_dev),
                        format!("Coefficient of variation: {:.2}", std_dev / mean),
                    ],
                    impact_score: 0.7,
                    urgency: InsightUrgency::Medium,
                });
            }

            // Performance degradation insight
            if values.len() >= 5 {
                let recent_avg = values.iter().rev().take(3).sum::<f64>() / 3.0;
                let historical_avg = values.iter().take(values.len() - 3).sum::<f64>() / (values.len() - 3) as f64;

                if recent_avg < historical_avg * 0.8 && metric_name.contains("throughput") {
                    insights.push(Insight {
                        category: "Performance".to_string(),
                        description: format!("Performance degradation detected in {}", metric_name),
                        confidence: 0.9,
                        supporting_data: vec![
                            format!("Recent average: {:.2}", recent_avg),
                            format!("Historical average: {:.2}", historical_avg),
                        ],
                        impact_score: 0.8,
                        urgency: InsightUrgency::High,
                    });
                }
            }
        }

        Ok(insights)
    }

    /// Analyze events for insights
    fn analyze_event_insights(&self, events: &[TaskExecutionEvent]) -> SklResult<Vec<Insight>> {
        let mut insights = Vec::new();

        if events.is_empty() {
            return Ok(insights);
        }

        // Analyze error patterns
        let error_events: Vec<&TaskExecutionEvent> = events.iter().filter(|e| e.is_error()).collect();
        let error_rate = error_events.len() as f64 / events.len() as f64;

        if error_rate > 0.1 {
            insights.push(Insight {
                category: "Reliability".to_string(),
                description: format!("High error rate detected: {:.1}%", error_rate * 100.0),
                confidence: 0.9,
                supporting_data: vec![
                    format!("Error events: {}", error_events.len()),
                    format!("Total events: {}", events.len()),
                ],
                impact_score: 0.8,
                urgency: InsightUrgency::High,
            });
        }

        // Analyze task duration patterns
        let durations: Vec<Duration> = events.iter()
            .filter_map(|e| e.duration())
            .collect();

        if !durations.is_empty() {
            let avg_duration = durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / durations.len() as f64;
            let long_tasks = durations.iter().filter(|d| d.as_secs_f64() > avg_duration * 2.0).count();

            if long_tasks as f64 / durations.len() as f64 > 0.2 {
                insights.push(Insight {
                    category: "Performance".to_string(),
                    description: "Significant number of long-running tasks detected".to_string(),
                    confidence: 0.8,
                    supporting_data: vec![
                        format!("Average duration: {:.2}s", avg_duration),
                        format!("Long tasks: {}", long_tasks),
                    ],
                    impact_score: 0.6,
                    urgency: InsightUrgency::Medium,
                });
            }
        }

        Ok(insights)
    }

    /// Generate trend insights
    fn generate_trend_insights(&self, trends: &PerformanceTrends) -> Vec<Insight> {
        let mut insights = Vec::new();

        if trends.throughput_trend == TrendDirection::Degrading {
            insights.push(Insight {
                category: "Trend".to_string(),
                description: "Throughput is degrading over time".to_string(),
                confidence: 0.8,
                supporting_data: vec!["Trend direction: Degrading".to_string()],
                impact_score: 0.7,
                urgency: InsightUrgency::Medium,
            });
        }

        if trends.error_rate_trend == TrendDirection::Improving {
            insights.push(Insight {
                category: "Trend".to_string(),
                description: "Error rate is improving over time".to_string(),
                confidence: 0.8,
                supporting_data: vec!["Trend direction: Improving".to_string()],
                impact_score: 0.5,
                urgency: InsightUrgency::Low,
            });
        }

        insights
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(&self, metrics: &[PerformanceMetric], events: &[TaskExecutionEvent], anomalies: &[Anomaly], patterns: &[Pattern]) -> SklResult<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Recommendations based on anomalies
        for anomaly in anomalies {
            if anomaly.severity == AnomalySeverity::Critical {
                recommendations.push(Recommendation {
                    category: "Performance".to_string(),
                    title: format!("Address critical anomaly in {}", anomaly.metric_name),
                    description: format!("Critical anomaly detected: {}", anomaly.description),
                    priority: RecommendationPriority::High,
                    expected_impact: anomaly.impact_score,
                    implementation_effort: ImplementationEffort::Medium,
                    action_items: vec![
                        "Investigate root cause of anomaly".to_string(),
                        "Implement corrective measures".to_string(),
                        "Monitor for recurrence".to_string(),
                    ],
                });
            }
        }

        // Recommendations based on error rate
        let error_events: Vec<&TaskExecutionEvent> = events.iter().filter(|e| e.is_error()).collect();
        if !events.is_empty() {
            let error_rate = error_events.len() as f64 / events.len() as f64;
            if error_rate > 0.05 {
                recommendations.push(Recommendation {
                    category: "Reliability".to_string(),
                    title: "Improve error handling and resilience".to_string(),
                    description: format!("High error rate detected: {:.1}%", error_rate * 100.0),
                    priority: RecommendationPriority::High,
                    expected_impact: 0.8,
                    implementation_effort: ImplementationEffort::High,
                    action_items: vec![
                        "Analyze error patterns and root causes".to_string(),
                        "Implement retry mechanisms".to_string(),
                        "Improve error monitoring".to_string(),
                        "Add circuit breakers for failing components".to_string(),
                    ],
                });
            }
        }

        // Recommendations based on patterns
        for pattern in patterns {
            if pattern.pattern_type == "resource_bottleneck" {
                recommendations.push(Recommendation {
                    category: "Scalability".to_string(),
                    title: "Address resource bottleneck".to_string(),
                    description: format!("Resource bottleneck pattern detected: {}", pattern.description),
                    priority: RecommendationPriority::Medium,
                    expected_impact: pattern.impact,
                    implementation_effort: ImplementationEffort::Medium,
                    action_items: vec![
                        "Scale affected resources".to_string(),
                        "Optimize resource utilization".to_string(),
                        "Implement load balancing".to_string(),
                    ],
                });
            }
        }

        Ok(recommendations)
    }

    /// Calculate confidence score for insights
    fn calculate_confidence_score(&self, anomalies: &[Anomaly], patterns: &[Pattern], correlations: &[Correlation]) -> f64 {
        let mut total_confidence = 0.0;
        let mut count = 0;

        for anomaly in anomalies {
            total_confidence += anomaly.confidence_score;
            count += 1;
        }

        for pattern in patterns {
            total_confidence += pattern.confidence;
            count += 1;
        }

        for correlation in correlations {
            total_confidence += correlation.strength.abs();
            count += 1;
        }

        if count > 0 {
            total_confidence / count as f64
        } else {
            1.0
        }
    }

    /// Generate cache key for insights
    fn generate_cache_key(&self, metrics: &[PerformanceMetric], events: &[TaskExecutionEvent]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        metrics.len().hash(&mut hasher);
        events.len().hash(&mut hasher);

        // Hash some representative data
        if !metrics.is_empty() {
            metrics[0].name.hash(&mut hasher);
            metrics[0].timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }

    /// Get cached insights
    fn get_cached_insights(&self, key: &str) -> Option<PerformanceInsights> {
        let cache = self.insights_cache.lock().unwrap();
        cache.get(key)
    }

    /// Cache insights
    fn cache_insights(&self, key: String, insights: &PerformanceInsights) {
        let mut cache = self.insights_cache.lock().unwrap();
        cache.insert(key, insights.clone());
    }

    /// Get analysis statistics
    pub fn statistics(&self) -> &AnalysisStatistics {
        &self.stats
    }
}

/// Anomaly detector trait
pub trait AnomalyDetector: Send + Sync {
    /// Detect anomalies in metrics
    fn detect_anomalies(&self, metrics: &[PerformanceMetric]) -> SklResult<Vec<Anomaly>>;

    /// Get detector name
    fn name(&self) -> &str;

    /// Get detector configuration
    fn config(&self) -> AnomalyDetectorConfig;
}

/// Trend analyzer trait
pub trait TrendAnalyzer: Send + Sync {
    /// Analyze trends in metrics
    fn analyze_trends(&self, metrics: &[PerformanceMetric]) -> SklResult<PerformanceTrends>;

    /// Get analyzer name
    fn name(&self) -> &str;
}

/// Pattern recognizer trait
pub trait PatternRecognizer: Send + Sync {
    /// Recognize patterns in metrics and events
    fn recognize_patterns(&self, metrics: &[PerformanceMetric], events: &[TaskExecutionEvent]) -> SklResult<Vec<Pattern>>;

    /// Get recognizer name
    fn name(&self) -> &str;
}

/// Correlation analyzer for detecting metric correlations
#[derive(Debug)]
pub struct CorrelationAnalyzer {
    /// Analysis configuration
    config: CorrelationAnalysisConfig,
}

impl CorrelationAnalyzer {
    /// Create new correlation analyzer
    pub fn new(config: CorrelationAnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze correlations between metrics
    pub fn analyze_correlations(&self, metrics: &[PerformanceMetric]) -> SklResult<Vec<Correlation>> {
        let mut correlations = Vec::new();

        // Group metrics by name
        let mut metric_groups: HashMap<String, Vec<f64>> = HashMap::new();
        for metric in metrics {
            metric_groups.entry(metric.name.clone()).or_insert_with(Vec::new).push(metric.value);
        }

        let metric_names: Vec<String> = metric_groups.keys().cloned().collect();

        // Calculate correlations between all pairs
        for i in 0..metric_names.len() {
            for j in i + 1..metric_names.len() {
                let name1 = &metric_names[i];
                let name2 = &metric_names[j];

                if let (Some(values1), Some(values2)) = (metric_groups.get(name1), metric_groups.get(name2)) {
                    if values1.len() == values2.len() && values1.len() > 1 {
                        let correlation_coefficient = self.calculate_correlation(values1, values2);

                        if correlation_coefficient.abs() >= self.config.min_correlation_threshold {
                            correlations.push(Correlation {
                                primary_metric: name1.clone(),
                                secondary_metric: name2.clone(),
                                coefficient: correlation_coefficient,
                                strength: correlation_coefficient.abs(),
                                correlation_type: if correlation_coefficient > 0.0 {
                                    CorrelationType::Positive
                                } else {
                                    CorrelationType::Negative
                                },
                                confidence: self.calculate_correlation_confidence(values1.len(), correlation_coefficient.abs()),
                            });
                        }
                    }
                }
            }
        }

        Ok(correlations)
    }

    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
        let sum_x2 = x.iter().map(|a| a * a).sum::<f64>();
        let sum_y2 = y.iter().map(|b| b * b).sum::<f64>();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate confidence in correlation
    fn calculate_correlation_confidence(&self, sample_size: usize, correlation_strength: f64) -> f64 {
        // Simple confidence calculation based on sample size and correlation strength
        let size_factor = (sample_size as f64 / 100.0).min(1.0);
        let strength_factor = correlation_strength;

        (size_factor * strength_factor).min(1.0)
    }
}

/// Performance insights containing analysis results
#[derive(Debug, Clone)]
pub struct PerformanceInsights {
    /// Generated insights
    pub insights: Vec<Insight>,

    /// Detected patterns
    pub patterns: Vec<Pattern>,

    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,

    /// Found correlations
    pub correlations: Vec<Correlation>,

    /// When insights were generated
    pub generated_at: SystemTime,

    /// Overall confidence score
    pub confidence_score: f64,

    /// Performance recommendations
    pub recommendations: Vec<Recommendation>,
}

/// Individual performance insight
#[derive(Debug, Clone)]
pub struct Insight {
    /// Insight category
    pub category: String,

    /// Insight description
    pub description: String,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Supporting data/evidence
    pub supporting_data: Vec<String>,

    /// Impact score (0.0 to 1.0)
    pub impact_score: f64,

    /// Insight urgency
    pub urgency: InsightUrgency,
}

/// Insight urgency levels
#[derive(Debug, Clone, PartialEq)]
pub enum InsightUrgency {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance pattern
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Pattern type
    pub pattern_type: String,

    /// Pattern description
    pub description: String,

    /// Pattern frequency
    pub frequency: String,

    /// Pattern impact (0.0 to 1.0)
    pub impact: f64,

    /// Pattern confidence (0.0 to 1.0)
    pub confidence: f64,

    /// Pattern occurrences
    pub occurrences: u32,

    /// First detected timestamp
    pub first_detected: SystemTime,

    /// Last detected timestamp
    pub last_detected: SystemTime,
}

/// Performance anomaly
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Anomaly type
    pub anomaly_type: String,

    /// Affected metric name
    pub metric_name: String,

    /// Detection timestamp
    pub detected_at: SystemTime,

    /// Anomaly score (0.0 to 1.0)
    pub anomaly_score: f64,

    /// Anomaly description
    pub description: String,

    /// Anomaly severity
    pub severity: AnomalySeverity,

    /// Confidence score (0.0 to 1.0)
    pub confidence_score: f64,

    /// Expected value
    pub expected_value: Option<f64>,

    /// Actual value
    pub actual_value: f64,

    /// Impact score (0.0 to 1.0)
    pub impact_score: f64,
}

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Metric correlation information
#[derive(Debug, Clone)]
pub struct Correlation {
    /// Primary metric name
    pub primary_metric: String,

    /// Secondary metric name
    pub secondary_metric: String,

    /// Correlation coefficient (-1.0 to 1.0)
    pub coefficient: f64,

    /// Correlation strength (0.0 to 1.0)
    pub strength: f64,

    /// Correlation type
    pub correlation_type: CorrelationType,

    /// Confidence in correlation (0.0 to 1.0)
    pub confidence: f64,
}

/// Types of correlations
#[derive(Debug, Clone, PartialEq)]
pub enum CorrelationType {
    Positive,
    Negative,
    NonLinear,
    Causal,
}

/// Performance recommendation
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Recommendation category
    pub category: String,

    /// Recommendation title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Expected impact (0.0 to 1.0)
    pub expected_impact: f64,

    /// Implementation effort
    pub implementation_effort: ImplementationEffort,

    /// Action items
    pub action_items: Vec<String>,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum RecommendationPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Implementation effort levels
#[derive(Debug, Clone, PartialEq)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Anomaly detector configuration
#[derive(Debug, Clone)]
pub struct AnomalyDetectorConfig {
    /// Sensitivity threshold (0.0 to 1.0)
    pub sensitivity: f64,

    /// Minimum sample size for detection
    pub min_sample_size: usize,

    /// Detection window size
    pub window_size: Duration,

    /// Enable seasonal adjustment
    pub seasonal_adjustment: bool,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            sensitivity: 0.8,
            min_sample_size: 10,
            window_size: Duration::from_secs(3600),
            seasonal_adjustment: false,
        }
    }
}

/// Correlation analysis configuration
#[derive(Debug, Clone)]
pub struct CorrelationAnalysisConfig {
    /// Minimum correlation threshold
    pub min_correlation_threshold: f64,

    /// Maximum metrics to analyze
    pub max_metrics: usize,

    /// Enable time-lagged correlations
    pub enable_time_lags: bool,

    /// Maximum time lag to consider
    pub max_time_lag: Duration,
}

impl Default for CorrelationAnalysisConfig {
    fn default() -> Self {
        Self {
            min_correlation_threshold: 0.7,
            max_metrics: 50,
            enable_time_lags: false,
            max_time_lag: Duration::from_secs(300),
        }
    }
}

/// Insights cache for performance optimization
#[derive(Debug)]
struct InsightsCache {
    cache: HashMap<String, (PerformanceInsights, SystemTime)>,
    max_size: usize,
    ttl: Duration,
}

impl InsightsCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_size: 100,
            ttl: Duration::from_secs(300), // 5 minutes
        }
    }

    fn get(&self, key: &str) -> Option<PerformanceInsights> {
        if let Some((insights, timestamp)) = self.cache.get(key) {
            if timestamp.elapsed().unwrap_or(Duration::from_secs(0)) < self.ttl {
                return Some(insights.clone());
            }
        }
        None
    }

    fn insert(&mut self, key: String, insights: &PerformanceInsights) {
        // Clean up expired entries
        let now = SystemTime::now();
        self.cache.retain(|_, (_, timestamp)| {
            timestamp.elapsed().unwrap_or(Duration::from_secs(0)) < self.ttl
        });

        // Enforce size limit
        if self.cache.len() >= self.max_size {
            // Remove oldest entry
            if let Some(oldest_key) = self.cache.iter()
                .min_by_key(|(_, (_, timestamp))| *timestamp)
                .map(|(k, _)| k.clone()) {
                self.cache.remove(&oldest_key);
            }
        }

        self.cache.insert(key, (insights.clone(), now));
    }
}

/// Analysis statistics
#[derive(Debug, Clone)]
pub struct AnalysisStatistics {
    /// Total analyses performed
    pub total_analyses: u64,

    /// Number of anomaly detections
    pub anomaly_detections: u64,

    /// Number of trend analyses
    pub trend_analyses: u64,

    /// Number of pattern recognitions
    pub pattern_recognitions: u64,

    /// Number of correlation analyses
    pub correlation_analyses: u64,

    /// Number of analysis failures
    pub analysis_failures: u64,

    /// Total analysis time
    pub total_analysis_time: Duration,

    /// Last analysis timestamp
    pub last_analysis: SystemTime,
}

impl AnalysisStatistics {
    fn new() -> Self {
        Self {
            total_analyses: 0,
            anomaly_detections: 0,
            trend_analyses: 0,
            pattern_recognitions: 0,
            correlation_analyses: 0,
            analysis_failures: 0,
            total_analysis_time: Duration::from_millis(0),
            last_analysis: SystemTime::now(),
        }
    }

    /// Calculate analysis success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.total_analyses + self.analysis_failures;
        if total > 0 {
            self.total_analyses as f64 / total as f64
        } else {
            1.0
        }
    }

    /// Calculate average analysis time
    pub fn avg_analysis_time(&self) -> Duration {
        if self.total_analyses > 0 {
            self.total_analysis_time / self.total_analyses as u32
        } else {
            Duration::from_millis(0)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_analyzer() {
        let config = CorrelationAnalysisConfig::default();
        let analyzer = CorrelationAnalyzer::new(config);

        // Create perfectly correlated metrics
        let metrics = vec![
            PerformanceMetric::new("metric_a".to_string(), 1.0, "value".to_string()),
            PerformanceMetric::new("metric_b".to_string(), 2.0, "value".to_string()),
            PerformanceMetric::new("metric_a".to_string(), 2.0, "value".to_string()),
            PerformanceMetric::new("metric_b".to_string(), 4.0, "value".to_string()),
            PerformanceMetric::new("metric_a".to_string(), 3.0, "value".to_string()),
            PerformanceMetric::new("metric_b".to_string(), 6.0, "value".to_string()),
        ];

        let correlations = analyzer.analyze_correlations(&metrics).unwrap();
        assert_eq!(correlations.len(), 1);

        let correlation = &correlations[0];
        assert_eq!(correlation.primary_metric, "metric_a");
        assert_eq!(correlation.secondary_metric, "metric_b");
        assert!(correlation.coefficient > 0.9); // Should be very high positive correlation
        assert_eq!(correlation.correlation_type, CorrelationType::Positive);
    }

    #[test]
    fn test_correlation_calculation() {
        let config = CorrelationAnalysisConfig::default();
        let analyzer = CorrelationAnalyzer::new(config);

        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = analyzer.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 0.001);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let correlation_neg = analyzer.calculate_correlation(&x, &y_neg);
        assert!((correlation_neg + 1.0).abs() < 0.001);

        // No correlation
        let y_random = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let correlation_none = analyzer.calculate_correlation(&x, &y_random);
        assert!(correlation_none.abs() < 0.8); // Should be relatively low
    }

    #[test]
    fn test_performance_insights_creation() {
        let insights = PerformanceInsights {
            insights: vec![
                Insight {
                    category: "Performance".to_string(),
                    description: "Test insight".to_string(),
                    confidence: 0.8,
                    supporting_data: vec!["test data".to_string()],
                    impact_score: 0.7,
                    urgency: InsightUrgency::Medium,
                }
            ],
            patterns: Vec::new(),
            anomalies: Vec::new(),
            correlations: Vec::new(),
            generated_at: SystemTime::now(),
            confidence_score: 0.8,
            recommendations: Vec::new(),
        };

        assert_eq!(insights.insights.len(), 1);
        assert_eq!(insights.confidence_score, 0.8);
    }

    #[test]
    fn test_anomaly_severity() {
        let anomaly = Anomaly {
            anomaly_type: "statistical".to_string(),
            metric_name: "cpu_usage".to_string(),
            detected_at: SystemTime::now(),
            anomaly_score: 0.9,
            description: "High CPU usage anomaly".to_string(),
            severity: AnomalySeverity::Critical,
            confidence_score: 0.85,
            expected_value: Some(0.3),
            actual_value: 0.95,
            impact_score: 0.8,
        };

        assert_eq!(anomaly.severity, AnomalySeverity::Critical);
        assert_eq!(anomaly.anomaly_score, 0.9);
        assert_eq!(anomaly.actual_value, 0.95);
    }

    #[test]
    fn test_recommendation_priority() {
        assert!(RecommendationPriority::Critical > RecommendationPriority::High);
        assert!(RecommendationPriority::High > RecommendationPriority::Medium);
        assert!(RecommendationPriority::Medium > RecommendationPriority::Low);
    }

    #[test]
    fn test_insights_cache() {
        let mut cache = InsightsCache::new();
        let insights = PerformanceInsights {
            insights: Vec::new(),
            patterns: Vec::new(),
            anomalies: Vec::new(),
            correlations: Vec::new(),
            generated_at: SystemTime::now(),
            confidence_score: 0.8,
            recommendations: Vec::new(),
        };

        // Test insertion and retrieval
        cache.insert("test_key".to_string(), &insights);
        let retrieved = cache.get("test_key");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().confidence_score, 0.8);

        // Test non-existent key
        let non_existent = cache.get("non_existent");
        assert!(non_existent.is_none());
    }

    #[test]
    fn test_analysis_statistics() {
        let mut stats = AnalysisStatistics::new();

        stats.total_analyses = 100;
        stats.analysis_failures = 5;

        assert_eq!(stats.success_rate(), 0.9523809523809523); // 100 / (100 + 5)

        stats.total_analysis_time = Duration::from_secs(50);
        assert_eq!(stats.avg_analysis_time(), Duration::from_millis(500)); // 50000ms / 100
    }
}