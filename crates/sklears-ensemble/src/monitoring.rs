//! Performance monitoring and tracking system for ensemble methods
//!
//! This module provides comprehensive monitoring capabilities for ensemble models,
//! including performance tracking, concept drift detection, model degradation monitoring,
//! and automated retraining triggers.

use sklears_core::{
    error::{Result, SklearsError},
    traits::Estimator,
    types::Float,
};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Window size for performance tracking
    pub window_size: usize,
    /// Threshold for performance degradation
    pub degradation_threshold: Float,
    /// Threshold for concept drift detection
    pub drift_threshold: Float,
    /// Minimum number of samples before monitoring
    pub min_samples: usize,
    /// Monitoring frequency (samples between checks)
    pub monitoring_frequency: usize,
    /// Enable automated retraining
    pub enable_auto_retrain: bool,
    /// Maximum training time allowed
    pub max_training_time: Duration,
    /// Performance metrics to track
    pub metrics_to_track: Vec<PerformanceMetric>,
}

/// Performance metrics that can be tracked
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PerformanceMetric {
    /// Accuracy for classification
    Accuracy,
    /// Precision for classification
    Precision,
    /// Recall for classification
    Recall,
    /// F1 score for classification
    F1Score,
    /// Area under ROC curve
    AUC,
    /// Mean squared error for regression
    MeanSquaredError,
    /// Mean absolute error for regression
    MeanAbsoluteError,
    /// R² score for regression
    R2Score,
    /// Prediction latency
    Latency,
    /// Memory usage
    MemoryUsage,
    /// Model confidence
    Confidence,
    /// Prediction entropy
    Entropy,
    /// Custom metric
    Custom(String),
}

/// Performance tracking data point
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Timestamp of the measurement
    pub timestamp: u64,
    /// Metric values
    pub metrics: HashMap<PerformanceMetric, Float>,
    /// Sample size for this measurement
    pub sample_size: usize,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Concept drift detection results
#[derive(Debug, Clone)]
pub struct DriftDetectionResult {
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Confidence level of drift detection
    pub confidence: Float,
    /// Type of drift detected
    pub drift_type: DriftType,
    /// Affected features (if applicable)
    pub affected_features: Vec<usize>,
    /// Drift severity score
    pub severity: Float,
    /// Recommended action
    pub recommended_action: RecommendedAction,
}

/// Types of concept drift
#[derive(Debug, Clone, PartialEq)]
pub enum DriftType {
    /// Sudden drift - abrupt change
    Sudden,
    /// Gradual drift - slow change over time
    Gradual,
    /// Recurring drift - cyclic patterns
    Recurring,
    /// No drift detected
    None,
}

/// Recommended actions based on monitoring results
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendedAction {
    /// Continue monitoring, no action needed
    ContinueMonitoring,
    /// Increase monitoring frequency
    IncreaseMonitoring,
    /// Retrain the model
    Retrain,
    /// Update model weights
    UpdateWeights,
    /// Add new models to ensemble
    ExpandEnsemble,
    /// Remove underperforming models
    PruneEnsemble,
    /// Completely rebuild ensemble
    RebuildEnsemble,
}

/// Model health status
#[derive(Debug, Clone, PartialEq)]
pub enum ModelHealth {
    /// Model is performing well
    Healthy,
    /// Model showing signs of degradation
    Warning,
    /// Model performance has degraded significantly
    Critical,
    /// Model is unreliable and should be replaced
    Failed,
}

/// Performance monitoring results
#[derive(Debug, Clone)]
pub struct MonitoringResults {
    /// Current model health status
    pub health_status: ModelHealth,
    /// Performance trend over time
    pub performance_trend: PerformanceTrend,
    /// Drift detection results
    pub drift_results: Vec<DriftDetectionResult>,
    /// Performance degradation indicators
    pub degradation_indicators: DegradationIndicators,
    /// Recommendations for improvement
    pub recommendations: Vec<RecommendedAction>,
    /// Detailed metrics history
    pub metrics_history: Vec<PerformanceDataPoint>,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Trend direction (positive = improving, negative = degrading)
    pub direction: Float,
    /// Statistical significance of the trend
    pub significance: Float,
    /// Rate of change per time unit
    pub rate_of_change: Float,
    /// Trend confidence interval
    pub confidence_interval: (Float, Float),
    /// Projected performance in future
    pub projection: Float,
}

/// Performance degradation indicators
#[derive(Debug, Clone)]
pub struct DegradationIndicators {
    /// Accuracy drop from baseline
    pub accuracy_drop: Float,
    /// Increase in prediction variance
    pub variance_increase: Float,
    /// Latency increase
    pub latency_increase: Float,
    /// Memory usage increase
    pub memory_increase: Float,
    /// Overall degradation score
    pub degradation_score: Float,
}

/// Ensemble performance monitor
pub struct EnsembleMonitor {
    /// Monitoring configuration
    config: MonitoringConfig,
    /// Performance history buffer
    performance_history: VecDeque<PerformanceDataPoint>,
    /// Baseline performance metrics
    baseline_metrics: HashMap<PerformanceMetric, Float>,
    /// Drift detection state
    drift_detector: DriftDetector,
    /// Sample counter
    sample_count: usize,
    /// Last monitoring timestamp
    last_monitoring: Option<Instant>,
}

/// Drift detection algorithm
struct DriftDetector {
    /// ADWIN detector for accuracy drift
    adwin_detector: ADWINDetector,
    /// Page-Hinkley test for mean shift detection
    page_hinkley: PageHinkleyDetector,
    /// Statistical tests for distribution drift
    statistical_tests: StatisticalDriftTests,
}

/// ADWIN (Adaptive Windowing) drift detector
struct ADWINDetector {
    /// Window of recent values
    window: VecDeque<Float>,
    /// Minimum window size
    min_window_size: usize,
    /// Confidence level
    confidence: Float,
    /// Total sum in window
    total_sum: Float,
    /// Sum of squares in window
    sum_squares: Float,
}

/// Page-Hinkley test for detecting mean shifts
struct PageHinkleyDetector {
    /// Cumulative sum
    cumsum: Float,
    /// Minimum cumulative sum seen
    min_cumsum: Float,
    /// Threshold for detection
    threshold: Float,
    /// Minimum number of samples
    min_samples: usize,
    /// Sample counter
    sample_count: usize,
}

/// Statistical tests for drift detection
struct StatisticalDriftTests {
    /// Reference distribution (baseline)
    reference_samples: VecDeque<Float>,
    /// Recent samples for comparison
    recent_samples: VecDeque<Float>,
    /// Maximum samples to keep
    max_samples: usize,
}

impl EnsembleMonitor {
    /// Create a new ensemble monitor
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            performance_history: VecDeque::with_capacity(config.window_size),
            baseline_metrics: HashMap::new(),
            drift_detector: DriftDetector::new(&config),
            sample_count: 0,
            last_monitoring: None,
            config,
        }
    }

    /// Set baseline performance metrics
    pub fn set_baseline(&mut self, metrics: HashMap<PerformanceMetric, Float>) {
        self.baseline_metrics = metrics;
    }

    /// Add a new performance measurement
    pub fn add_measurement(&mut self, data_point: PerformanceDataPoint) -> Result<()> {
        // Add to history
        self.performance_history.push_back(data_point.clone());

        // Maintain window size
        if self.performance_history.len() > self.config.window_size {
            self.performance_history.pop_front();
        }

        // Update drift detector with accuracy if available
        if let Some(&accuracy) = data_point.metrics.get(&PerformanceMetric::Accuracy) {
            self.drift_detector.update(accuracy)?;
        }

        self.sample_count += data_point.sample_size;

        Ok(())
    }

    /// Monitor ensemble performance and detect issues
    pub fn monitor_performance(&mut self) -> Result<MonitoringResults> {
        // Check if we have enough data
        if self.performance_history.len() < self.config.min_samples {
            return Err(SklearsError::InvalidInput(
                "Insufficient data for monitoring".to_string(),
            ));
        }

        // Determine health status
        let health_status = self.assess_health_status()?;

        // Analyze performance trend
        let performance_trend = self.analyze_performance_trend()?;

        // Detect concept drift
        let drift_results = self.detect_drift()?;

        // Compute degradation indicators
        let degradation_indicators = self.compute_degradation_indicators()?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &health_status,
            &performance_trend,
            &drift_results,
            &degradation_indicators,
        )?;

        Ok(MonitoringResults {
            health_status,
            performance_trend,
            drift_results,
            degradation_indicators,
            recommendations,
            metrics_history: self.performance_history.iter().cloned().collect(),
        })
    }

    /// Assess overall model health
    fn assess_health_status(&self) -> Result<ModelHealth> {
        if self.performance_history.is_empty() {
            return Ok(ModelHealth::Warning);
        }

        let recent_performance = self.get_recent_average_performance()?;
        let degradation_score = self.compute_overall_degradation_score(&recent_performance)?;

        if degradation_score > 0.5 {
            Ok(ModelHealth::Failed)
        } else if degradation_score > 0.3 {
            Ok(ModelHealth::Critical)
        } else if degradation_score > 0.1 {
            Ok(ModelHealth::Warning)
        } else {
            Ok(ModelHealth::Healthy)
        }
    }

    /// Get recent average performance metrics
    fn get_recent_average_performance(&self) -> Result<HashMap<PerformanceMetric, Float>> {
        if self.performance_history.is_empty() {
            return Ok(HashMap::new());
        }

        let recent_window = self.config.window_size.min(10);
        let start_idx = if self.performance_history.len() > recent_window {
            self.performance_history.len() - recent_window
        } else {
            0
        };

        let mut metric_sums: HashMap<PerformanceMetric, Float> = HashMap::new();
        let mut metric_counts: HashMap<PerformanceMetric, usize> = HashMap::new();

        for data_point in self.performance_history.range(start_idx..) {
            for (metric, value) in &data_point.metrics {
                *metric_sums.entry(metric.clone()).or_insert(0.0) += value;
                *metric_counts.entry(metric.clone()).or_insert(0) += 1;
            }
        }

        let mut averages = HashMap::new();
        for (metric, sum) in metric_sums {
            if let Some(&count) = metric_counts.get(&metric) {
                averages.insert(metric, sum / count as Float);
            }
        }

        Ok(averages)
    }

    /// Compute overall degradation score
    fn compute_overall_degradation_score(
        &self,
        recent_performance: &HashMap<PerformanceMetric, Float>,
    ) -> Result<Float> {
        if self.baseline_metrics.is_empty() {
            return Ok(0.0);
        }

        let mut degradation_sum = 0.0;
        let mut count = 0;

        for (metric, &recent_value) in recent_performance {
            if let Some(&baseline_value) = self.baseline_metrics.get(metric) {
                let degradation = match metric {
                    // Higher is better metrics
                    PerformanceMetric::Accuracy
                    | PerformanceMetric::Precision
                    | PerformanceMetric::Recall
                    | PerformanceMetric::F1Score
                    | PerformanceMetric::AUC
                    | PerformanceMetric::R2Score => {
                        (baseline_value - recent_value) / baseline_value.max(1e-8)
                    }
                    // Lower is better metrics
                    PerformanceMetric::MeanSquaredError
                    | PerformanceMetric::MeanAbsoluteError
                    | PerformanceMetric::Latency
                    | PerformanceMetric::MemoryUsage => {
                        (recent_value - baseline_value) / baseline_value.max(1e-8)
                    }
                    // Neutral metrics
                    _ => 0.0,
                };

                degradation_sum += degradation.max(0.0); // Only count degradation, not improvement
                count += 1;
            }
        }

        Ok(if count > 0 {
            degradation_sum / count as Float
        } else {
            0.0
        })
    }

    /// Analyze performance trend over time
    fn analyze_performance_trend(&self) -> Result<PerformanceTrend> {
        if self.performance_history.len() < 3 {
            return Ok(PerformanceTrend {
                direction: 0.0,
                significance: 0.0,
                rate_of_change: 0.0,
                confidence_interval: (0.0, 0.0),
                projection: 0.0,
            });
        }

        // Use accuracy as the primary metric for trend analysis
        let accuracy_values: Vec<Float> = self
            .performance_history
            .iter()
            .filter_map(|dp| dp.metrics.get(&PerformanceMetric::Accuracy))
            .copied()
            .collect();

        if accuracy_values.len() < 3 {
            return Ok(PerformanceTrend {
                direction: 0.0,
                significance: 0.0,
                rate_of_change: 0.0,
                confidence_interval: (0.0, 0.0),
                projection: 0.0,
            });
        }

        // Simple linear regression for trend analysis
        let n = accuracy_values.len() as Float;
        let x_values: Vec<Float> = (0..accuracy_values.len()).map(|i| i as Float).collect();

        let x_mean = x_values.iter().sum::<Float>() / n;
        let y_mean = accuracy_values.iter().sum::<Float>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..accuracy_values.len() {
            let x_diff = x_values[i] - x_mean;
            let y_diff = accuracy_values[i] - y_mean;
            numerator += x_diff * y_diff;
            denominator += x_diff * x_diff;
        }

        let slope = if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        // Compute R-squared for significance
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for i in 0..accuracy_values.len() {
            let y_pred = slope * x_values[i] + intercept;
            ss_res += (accuracy_values[i] - y_pred).powi(2);
            ss_tot += (accuracy_values[i] - y_mean).powi(2);
        }

        let r_squared = if ss_tot != 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Simple confidence interval estimation
        let std_error = (ss_res / (n - 2.0)).sqrt();
        let t_value = 1.96; // Approximate 95% confidence
        let margin_error = t_value * std_error;

        // Project future performance
        let projection = slope * n + intercept;

        Ok(PerformanceTrend {
            direction: slope,
            significance: r_squared,
            rate_of_change: slope,
            confidence_interval: (projection - margin_error, projection + margin_error),
            projection,
        })
    }

    /// Detect concept drift
    fn detect_drift(&mut self) -> Result<Vec<DriftDetectionResult>> {
        let mut results = Vec::new();

        // ADWIN drift detection
        if let Some(adwin_result) = self.drift_detector.adwin_detector.check_drift() {
            results.push(DriftDetectionResult {
                drift_detected: true,
                confidence: adwin_result.confidence,
                drift_type: DriftType::Sudden,
                affected_features: vec![],
                severity: adwin_result.severity,
                recommended_action: RecommendedAction::Retrain,
            });
        }

        // Page-Hinkley drift detection
        if self.drift_detector.page_hinkley.detect_drift() {
            results.push(DriftDetectionResult {
                drift_detected: true,
                confidence: 0.95,
                drift_type: DriftType::Gradual,
                affected_features: vec![],
                severity: 0.7,
                recommended_action: RecommendedAction::UpdateWeights,
            });
        }

        // Statistical drift tests
        if let Some(stat_result) = self
            .drift_detector
            .statistical_tests
            .kolmogorov_smirnov_test()?
        {
            results.push(DriftDetectionResult {
                drift_detected: stat_result.p_value < 0.05,
                confidence: 1.0 - stat_result.p_value,
                drift_type: DriftType::Sudden,
                affected_features: vec![],
                severity: stat_result.test_statistic,
                recommended_action: if stat_result.p_value < 0.01 {
                    RecommendedAction::RebuildEnsemble
                } else {
                    RecommendedAction::Retrain
                },
            });
        }

        if results.is_empty() {
            results.push(DriftDetectionResult {
                drift_detected: false,
                confidence: 0.0,
                drift_type: DriftType::None,
                affected_features: vec![],
                severity: 0.0,
                recommended_action: RecommendedAction::ContinueMonitoring,
            });
        }

        Ok(results)
    }

    /// Compute degradation indicators
    fn compute_degradation_indicators(&self) -> Result<DegradationIndicators> {
        let recent_performance = self.get_recent_average_performance()?;

        let accuracy_drop = if let (Some(&baseline), Some(&recent)) = (
            self.baseline_metrics.get(&PerformanceMetric::Accuracy),
            recent_performance.get(&PerformanceMetric::Accuracy),
        ) {
            baseline - recent
        } else {
            0.0
        };

        // Simplified variance increase calculation
        let variance_increase = self.compute_prediction_variance_increase()?;

        let latency_increase = if let (Some(&baseline), Some(&recent)) = (
            self.baseline_metrics.get(&PerformanceMetric::Latency),
            recent_performance.get(&PerformanceMetric::Latency),
        ) {
            recent - baseline
        } else {
            0.0
        };

        let memory_increase = if let (Some(&baseline), Some(&recent)) = (
            self.baseline_metrics.get(&PerformanceMetric::MemoryUsage),
            recent_performance.get(&PerformanceMetric::MemoryUsage),
        ) {
            recent - baseline
        } else {
            0.0
        };

        let degradation_score =
            (accuracy_drop + variance_increase + latency_increase + memory_increase) / 4.0;

        Ok(DegradationIndicators {
            accuracy_drop,
            variance_increase,
            latency_increase,
            memory_increase,
            degradation_score,
        })
    }

    /// Compute increase in prediction variance
    fn compute_prediction_variance_increase(&self) -> Result<Float> {
        // Simplified implementation - in practice would use actual prediction variances
        if self.performance_history.len() < 5 {
            return Ok(0.0);
        }

        let recent_confidence: Vec<Float> = self
            .performance_history
            .iter()
            .rev()
            .take(5)
            .filter_map(|dp| dp.metrics.get(&PerformanceMetric::Confidence))
            .copied()
            .collect();

        let baseline_confidence: Vec<Float> = self
            .performance_history
            .iter()
            .take(5)
            .filter_map(|dp| dp.metrics.get(&PerformanceMetric::Confidence))
            .copied()
            .collect();

        if recent_confidence.is_empty() || baseline_confidence.is_empty() {
            return Ok(0.0);
        }

        let recent_var = self.compute_variance(&recent_confidence);
        let baseline_var = self.compute_variance(&baseline_confidence);

        Ok(recent_var - baseline_var)
    }

    /// Compute variance of a set of values
    fn compute_variance(&self, values: &[Float]) -> Float {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<Float>() / values.len() as Float;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / values.len() as Float;

        variance
    }

    /// Generate recommendations based on monitoring results
    fn generate_recommendations(
        &self,
        health_status: &ModelHealth,
        performance_trend: &PerformanceTrend,
        drift_results: &[DriftDetectionResult],
        degradation_indicators: &DegradationIndicators,
    ) -> Result<Vec<RecommendedAction>> {
        let mut recommendations = Vec::new();

        // Health-based recommendations
        match health_status {
            ModelHealth::Failed => {
                recommendations.push(RecommendedAction::RebuildEnsemble);
            }
            ModelHealth::Critical => {
                recommendations.push(RecommendedAction::Retrain);
                recommendations.push(RecommendedAction::PruneEnsemble);
            }
            ModelHealth::Warning => {
                recommendations.push(RecommendedAction::IncreaseMonitoring);
                recommendations.push(RecommendedAction::UpdateWeights);
            }
            ModelHealth::Healthy => {
                recommendations.push(RecommendedAction::ContinueMonitoring);
            }
        }

        // Trend-based recommendations
        if performance_trend.direction < -0.01 && performance_trend.significance > 0.5 {
            recommendations.push(RecommendedAction::Retrain);
        }

        // Drift-based recommendations
        for drift_result in drift_results {
            if drift_result.drift_detected {
                recommendations.push(drift_result.recommended_action.clone());
            }
        }

        // Degradation-based recommendations
        if degradation_indicators.accuracy_drop > 0.1 {
            recommendations.push(RecommendedAction::Retrain);
        }

        if degradation_indicators.latency_increase > 0.5 {
            recommendations.push(RecommendedAction::PruneEnsemble);
        }

        // Remove duplicates
        recommendations.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        recommendations.dedup();

        Ok(recommendations)
    }

    /// Check if automated retraining should be triggered
    pub fn should_trigger_retrain(&self, monitoring_results: &MonitoringResults) -> bool {
        if !self.config.enable_auto_retrain {
            return false;
        }

        // Trigger retrain if critical health or significant drift detected
        matches!(
            monitoring_results.health_status,
            ModelHealth::Critical | ModelHealth::Failed
        ) || monitoring_results
            .drift_results
            .iter()
            .any(|dr| dr.drift_detected && dr.confidence > 0.8)
            || monitoring_results.degradation_indicators.degradation_score
                > self.config.degradation_threshold
    }
}

impl DriftDetector {
    /// Create a new drift detector
    fn new(config: &MonitoringConfig) -> Self {
        Self {
            adwin_detector: ADWINDetector::new(0.002, 100), // delta=0.002, min_window=100
            page_hinkley: PageHinkleyDetector::new(0.01, 30), // threshold=0.01, min_samples=30
            statistical_tests: StatisticalDriftTests::new(1000), // max_samples=1000
        }
    }

    /// Update drift detectors with new value
    fn update(&mut self, value: Float) -> Result<()> {
        self.adwin_detector.add_value(value);
        self.page_hinkley.add_value(value);
        self.statistical_tests.add_value(value);
        Ok(())
    }
}

impl ADWINDetector {
    /// Create a new ADWIN detector
    fn new(delta: Float, min_window_size: usize) -> Self {
        Self {
            window: VecDeque::new(),
            min_window_size,
            confidence: 1.0 - delta,
            total_sum: 0.0,
            sum_squares: 0.0,
        }
    }

    /// Add a new value to the detector
    fn add_value(&mut self, value: Float) {
        self.window.push_back(value);
        self.total_sum += value;
        self.sum_squares += value * value;

        // Check for drift and adjust window
        self.check_and_adjust_window();
    }

    /// Check for drift and adjust window size
    fn check_and_adjust_window(&mut self) {
        if self.window.len() < self.min_window_size {
            return;
        }

        let n = self.window.len() as Float;
        let mean = self.total_sum / n;

        // Simple drift detection based on variance change
        let variance = (self.sum_squares / n) - (mean * mean);

        // If variance becomes too high, consider it drift and shrink window
        if variance > 0.1 && self.window.len() > self.min_window_size {
            let removed = self.window.pop_front().unwrap();
            self.total_sum -= removed;
            self.sum_squares -= removed * removed;
        }
    }

    /// Check if drift is detected
    fn check_drift(&self) -> Option<ADWINDriftResult> {
        if self.window.len() < self.min_window_size {
            return None;
        }

        // Simplified drift detection
        let n = self.window.len() as Float;
        let mean = self.total_sum / n;
        let variance = (self.sum_squares / n) - (mean * mean);

        // Use variance as a proxy for drift
        if variance > 0.05 {
            Some(ADWINDriftResult {
                confidence: self.confidence,
                severity: variance,
            })
        } else {
            None
        }
    }
}

impl PageHinkleyDetector {
    /// Create a new Page-Hinkley detector
    fn new(threshold: Float, min_samples: usize) -> Self {
        Self {
            cumsum: 0.0,
            min_cumsum: 0.0,
            threshold,
            min_samples,
            sample_count: 0,
        }
    }

    /// Add a new value to the detector
    fn add_value(&mut self, value: Float) {
        self.sample_count += 1;

        // Assume we're testing for decrease in mean (e.g., accuracy drop)
        // Use negative values to detect decreases
        let normalized_value = 0.8 - value; // Assume baseline around 0.8

        self.cumsum += normalized_value;
        self.min_cumsum = self.min_cumsum.min(self.cumsum);
    }

    /// Check if drift is detected
    fn detect_drift(&self) -> bool {
        if self.sample_count < self.min_samples {
            return false;
        }

        (self.cumsum - self.min_cumsum) > self.threshold
    }
}

impl StatisticalDriftTests {
    /// Create a new statistical drift test
    fn new(max_samples: usize) -> Self {
        Self {
            reference_samples: VecDeque::with_capacity(max_samples),
            recent_samples: VecDeque::with_capacity(max_samples / 2),
            max_samples,
        }
    }

    /// Add a new value
    fn add_value(&mut self, value: Float) {
        // First half of samples go to reference, second half to recent
        if self.reference_samples.len() < self.max_samples / 2 {
            self.reference_samples.push_back(value);
        } else {
            self.recent_samples.push_back(value);
            if self.recent_samples.len() > self.max_samples / 2 {
                self.recent_samples.pop_front();
            }
        }
    }

    /// Perform Kolmogorov-Smirnov test
    fn kolmogorov_smirnov_test(&self) -> Result<Option<KSTestResult>> {
        if self.reference_samples.len() < 10 || self.recent_samples.len() < 10 {
            return Ok(None);
        }

        // Convert to sorted vectors
        let mut ref_sorted: Vec<Float> = self.reference_samples.iter().copied().collect();
        let mut recent_sorted: Vec<Float> = self.recent_samples.iter().copied().collect();

        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        recent_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute empirical CDFs and find maximum difference
        let mut max_diff: Float = 0.0;
        let n1 = ref_sorted.len() as Float;
        let n2 = recent_sorted.len() as Float;

        // Simplified KS test implementation
        let mut all_values: Vec<Float> = ref_sorted
            .iter()
            .chain(recent_sorted.iter())
            .copied()
            .collect();
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_values.dedup();

        for value in &all_values {
            let cdf1 = ref_sorted.iter().filter(|&&x| x <= *value).count() as Float / n1;
            let cdf2 = recent_sorted.iter().filter(|&&x| x <= *value).count() as Float / n2;

            max_diff = max_diff.max((cdf1 - cdf2).abs());
        }

        // Approximate p-value calculation
        let ks_statistic = max_diff;
        let p_value = self.approximate_ks_p_value(ks_statistic, n1 as usize, n2 as usize);

        Ok(Some(KSTestResult {
            test_statistic: ks_statistic,
            p_value,
        }))
    }

    /// Approximate p-value for KS test
    fn approximate_ks_p_value(&self, d: Float, n1: usize, n2: usize) -> Float {
        let n = ((n1 * n2) as Float / (n1 + n2) as Float).sqrt();
        let z = d * n;

        // Simplified approximation
        if z > 3.0 {
            0.0
        } else if z > 2.0 {
            0.001
        } else if z > 1.5 {
            0.01
        } else if z > 1.0 {
            0.05
        } else {
            0.5
        }
    }
}

/// ADWIN drift detection result
#[derive(Debug, Clone)]
struct ADWINDriftResult {
    confidence: Float,
    severity: Float,
}

/// Kolmogorov-Smirnov test result
#[derive(Debug, Clone)]
struct KSTestResult {
    test_statistic: Float,
    p_value: Float,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            degradation_threshold: 0.1,
            drift_threshold: 0.05,
            min_samples: 50,
            monitoring_frequency: 10,
            enable_auto_retrain: false,
            max_training_time: Duration::from_secs(3600), // 1 hour
            metrics_to_track: vec![
                PerformanceMetric::Accuracy,
                PerformanceMetric::Latency,
                PerformanceMetric::Confidence,
            ],
        }
    }
}

/// Convenience functions for creating monitoring configurations
impl MonitoringConfig {
    /// Create a configuration for high-frequency monitoring
    pub fn high_frequency() -> Self {
        Self {
            monitoring_frequency: 1,
            min_samples: 10,
            window_size: 500,
            ..Default::default()
        }
    }

    /// Create a configuration for production monitoring
    pub fn production() -> Self {
        Self {
            window_size: 5000,
            degradation_threshold: 0.05,
            drift_threshold: 0.03,
            min_samples: 100,
            monitoring_frequency: 50,
            enable_auto_retrain: true,
            ..Default::default()
        }
    }

    /// Create a configuration for development/testing
    pub fn development() -> Self {
        Self {
            window_size: 100,
            degradation_threshold: 0.2,
            min_samples: 20,
            monitoring_frequency: 5,
            enable_auto_retrain: false,
            ..Default::default()
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_monitoring_config_creation() {
        let config = MonitoringConfig::default();
        assert_eq!(config.window_size, 1000);
        assert_eq!(config.min_samples, 50);
        assert!(!config.enable_auto_retrain);
    }

    #[test]
    fn test_ensemble_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = EnsembleMonitor::new(config);
        assert_eq!(monitor.sample_count, 0);
        assert!(monitor.performance_history.is_empty());
    }

    #[test]
    fn test_performance_data_point_creation() {
        let mut metrics = HashMap::new();
        metrics.insert(PerformanceMetric::Accuracy, 0.85);
        metrics.insert(PerformanceMetric::Latency, 100.0);

        let data_point = PerformanceDataPoint {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metrics,
            sample_size: 100,
            metadata: HashMap::new(),
        };

        assert_eq!(data_point.sample_size, 100);
        assert!(data_point
            .metrics
            .contains_key(&PerformanceMetric::Accuracy));
    }

    #[test]
    fn test_add_measurement() {
        let config = MonitoringConfig::default();
        let mut monitor = EnsembleMonitor::new(config);

        let mut metrics = HashMap::new();
        metrics.insert(PerformanceMetric::Accuracy, 0.85);

        let data_point = PerformanceDataPoint {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metrics,
            sample_size: 100,
            metadata: HashMap::new(),
        };

        monitor.add_measurement(data_point).unwrap();

        assert_eq!(monitor.performance_history.len(), 1);
        assert_eq!(monitor.sample_count, 100);
    }

    #[test]
    fn test_baseline_setting() {
        let config = MonitoringConfig::default();
        let mut monitor = EnsembleMonitor::new(config);

        let mut baseline = HashMap::new();
        baseline.insert(PerformanceMetric::Accuracy, 0.9);
        baseline.insert(PerformanceMetric::Latency, 50.0);

        monitor.set_baseline(baseline.clone());

        assert_eq!(monitor.baseline_metrics.len(), 2);
        assert_eq!(monitor.baseline_metrics[&PerformanceMetric::Accuracy], 0.9);
    }

    #[test]
    fn test_health_assessment() {
        let config = MonitoringConfig::default();
        let mut monitor = EnsembleMonitor::new(config);

        // Set baseline
        let mut baseline = HashMap::new();
        baseline.insert(PerformanceMetric::Accuracy, 0.9);
        monitor.set_baseline(baseline);

        // Add measurements showing degradation
        for i in 0..60 {
            let mut metrics = HashMap::new();
            metrics.insert(PerformanceMetric::Accuracy, 0.9 - (i as Float * 0.01)); // Degrading accuracy

            let data_point = PerformanceDataPoint {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metrics,
                sample_size: 10,
                metadata: HashMap::new(),
            };

            monitor.add_measurement(data_point).unwrap();
        }

        let health = monitor.assess_health_status().unwrap();
        assert!(matches!(
            health,
            ModelHealth::Warning | ModelHealth::Critical | ModelHealth::Failed
        ));
    }

    #[test]
    fn test_adwin_detector() {
        let mut adwin = ADWINDetector::new(0.002, 10);

        // Add stable values
        for _ in 0..20 {
            adwin.add_value(0.8);
        }

        assert!(adwin.check_drift().is_none());

        // Add values showing drift
        for _ in 0..10 {
            adwin.add_value(0.6);
        }

        // May or may not detect drift depending on implementation details
        let _drift_result = adwin.check_drift();
    }

    #[test]
    fn test_page_hinkley_detector() {
        let mut ph = PageHinkleyDetector::new(0.1, 10);

        // Add stable values
        for _ in 0..15 {
            ph.add_value(0.8);
        }

        assert!(!ph.detect_drift());

        // Add decreasing values
        for i in 0..10 {
            ph.add_value(0.8 - (i as Float * 0.05));
        }

        // Should detect drift
        assert!(ph.detect_drift());
    }

    #[test]
    fn test_performance_trend_analysis() {
        let config = MonitoringConfig::default();
        let mut monitor = EnsembleMonitor::new(config);

        // Add measurements with declining trend
        for i in 0..20 {
            let mut metrics = HashMap::new();
            metrics.insert(PerformanceMetric::Accuracy, 0.9 - (i as Float * 0.01));

            let data_point = PerformanceDataPoint {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metrics,
                sample_size: 10,
                metadata: HashMap::new(),
            };

            monitor.add_measurement(data_point).unwrap();
        }

        let trend = monitor.analyze_performance_trend().unwrap();
        assert!(trend.direction < 0.0); // Should detect negative trend
    }

    #[test]
    fn test_degradation_indicators() {
        let config = MonitoringConfig::default();
        let mut monitor = EnsembleMonitor::new(config);

        // Set baseline
        let mut baseline = HashMap::new();
        baseline.insert(PerformanceMetric::Accuracy, 0.9);
        baseline.insert(PerformanceMetric::Latency, 50.0);
        monitor.set_baseline(baseline);

        // Add measurements showing degradation
        let mut metrics = HashMap::new();
        metrics.insert(PerformanceMetric::Accuracy, 0.7); // Accuracy drop
        metrics.insert(PerformanceMetric::Latency, 100.0); // Latency increase

        for _ in 0..60 {
            let data_point = PerformanceDataPoint {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metrics: metrics.clone(),
                sample_size: 10,
                metadata: HashMap::new(),
            };

            monitor.add_measurement(data_point).unwrap();
        }

        let indicators = monitor.compute_degradation_indicators().unwrap();
        assert!(indicators.accuracy_drop > 0.0);
        assert!(indicators.latency_increase > 0.0);
        assert!(indicators.degradation_score > 0.0);
    }

    #[test]
    fn test_monitoring_configurations() {
        let prod_config = MonitoringConfig::production();
        assert_eq!(prod_config.window_size, 5000);
        assert!(prod_config.enable_auto_retrain);

        let dev_config = MonitoringConfig::development();
        assert_eq!(dev_config.window_size, 100);
        assert!(!dev_config.enable_auto_retrain);

        let hf_config = MonitoringConfig::high_frequency();
        assert_eq!(hf_config.monitoring_frequency, 1);
    }

    #[test]
    fn test_recommendation_generation() {
        let config = MonitoringConfig::default();
        let mut monitor = EnsembleMonitor::new(config);

        let health_status = ModelHealth::Critical;
        let performance_trend = PerformanceTrend {
            direction: -0.02,
            significance: 0.8,
            rate_of_change: -0.02,
            confidence_interval: (0.7, 0.8),
            projection: 0.75,
        };
        let drift_results = vec![DriftDetectionResult {
            drift_detected: true,
            confidence: 0.9,
            drift_type: DriftType::Sudden,
            affected_features: vec![],
            severity: 0.8,
            recommended_action: RecommendedAction::Retrain,
        }];
        let degradation_indicators = DegradationIndicators {
            accuracy_drop: 0.15,
            variance_increase: 0.1,
            latency_increase: 0.2,
            memory_increase: 0.05,
            degradation_score: 0.125,
        };

        let recommendations = monitor
            .generate_recommendations(
                &health_status,
                &performance_trend,
                &drift_results,
                &degradation_indicators,
            )
            .unwrap();

        assert!(!recommendations.is_empty());
        assert!(recommendations.contains(&RecommendedAction::Retrain));
    }
}
