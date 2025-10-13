//! Temporal and Dynamic Metrics for Machine Learning
//!
//! This module provides comprehensive temporal analysis capabilities for machine learning
//! metrics, including concept drift detection, time-varying analysis, and adaptive weighting.
//! These metrics are essential for monitoring model performance over time and detecting
//! changes in data distribution that may require model retraining.
//!
//! # Features
//!
//! - Concept drift detection using statistical tests and sliding windows
//! - Time-varying metric analysis with trend detection
//! - Adaptive metric weighting based on temporal patterns
//! - Temporal stability measures for metric reliability assessment
//! - Metric evolution tracking with change point detection
//! - Seasonal decomposition and trend analysis
//! - Rolling statistics and temporal aggregations
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::temporal::*;
//! use scirs2_core::ndarray::Array1;
//!
//! // Create temporal metric analyzer
//! let mut analyzer = TemporalMetricsAnalyzer::new(WindowConfig::default());
//!
//! // Add time series of accuracy measurements (more data points for drift detection)
//! let accuracy_series = vec![0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.85, 0.82, 0.80, 0.78, 0.75, 0.73, 0.70, 0.68, 0.65];
//! for (i, &acc) in accuracy_series.iter().enumerate() {
//!     analyzer.add_measurement("accuracy", i as f64, acc);
//! }
//!
//! // Detect concept drift (handle case where insufficient data)
//! match analyzer.detect_concept_drift("accuracy", DriftDetectionMethod::KSTest) {
//!     Ok(drift_result) => println!("Drift detected: {}", drift_result.drift_detected),
//!     Err(_) => println!("Insufficient data for drift detection"),
//! }
//!
//! // Analyze temporal trends (handle case where insufficient data)
//! match analyzer.analyze_temporal_trend("accuracy") {
//!     Ok(trend_analysis) => println!("Trend slope: {:.4}", trend_analysis.slope),
//!     Err(_) => println!("Insufficient data for trend analysis"),
//! }
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;
use std::collections::{HashMap, VecDeque};

/// Configuration for temporal analysis windows
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Size of the sliding window for drift detection
    pub window_size: usize,
    /// Minimum number of samples required for analysis
    pub min_samples: usize,
    /// Overlap between consecutive windows
    pub overlap: f64,
    /// Significance level for statistical tests
    pub significance_level: f64,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            min_samples: 20,
            overlap: 0.5,
            significance_level: 0.05,
        }
    }
}

/// Methods for concept drift detection
#[derive(Debug, Clone, Copy)]
pub enum DriftDetectionMethod {
    /// Kolmogorov-Smirnov test
    KSTest,
    /// Page-Hinkley test
    PageHinkley,
    /// ADWIN (Adaptive Windowing)
    ADWIN,
    /// Statistical Process Control
    SPC,
    /// Population Stability Index
    PSI,
}

/// Result of concept drift detection
#[derive(Debug, Clone)]
pub struct ConceptDriftResult {
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value of the test (if applicable)
    pub p_value: Option<f64>,
    /// Confidence level of detection
    pub confidence: f64,
    /// Time point where drift was detected
    pub drift_point: Option<f64>,
    /// Magnitude of the drift
    pub drift_magnitude: f64,
    /// Detection method used
    pub method: DriftDetectionMethod,
}

/// Result of temporal trend analysis
#[derive(Debug, Clone)]
pub struct TemporalTrendAnalysis {
    /// Linear trend slope
    pub slope: f64,
    /// Intercept of the trend line
    pub intercept: f64,
    /// R-squared of the trend fit
    pub r_squared: f64,
    /// Statistical significance of the trend
    pub p_value: f64,
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Seasonal component (if detected)
    pub seasonal_component: Option<SeasonalComponent>,
    /// Change points detected
    pub change_points: Vec<f64>,
}

/// Direction of temporal trend
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendDirection {
    /// Increasing
    Increasing,
    /// Decreasing
    Decreasing,
    /// Stable
    Stable,
    /// Volatile
    Volatile,
}

/// Seasonal component analysis
#[derive(Debug, Clone)]
pub struct SeasonalComponent {
    /// Period of the seasonality
    pub period: f64,
    /// Amplitude of seasonal variation
    pub amplitude: f64,
    /// Phase shift
    pub phase: f64,
    /// Strength of seasonality (0-1)
    pub strength: f64,
}

/// Time series data point
#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    pub timestamp: f64,
    pub value: f64,
    pub metadata: HashMap<String, f64>,
}

/// Temporal metrics analyzer
pub struct TemporalMetricsAnalyzer {
    config: WindowConfig,
    time_series: HashMap<String, Vec<TimeSeriesPoint>>,
    _drift_detectors: HashMap<String, Box<dyn DriftDetector>>,
}

impl TemporalMetricsAnalyzer {
    /// Create new temporal analyzer
    pub fn new(config: WindowConfig) -> Self {
        Self {
            config,
            time_series: HashMap::new(),
            _drift_detectors: HashMap::new(),
        }
    }

    /// Add a measurement to the time series
    pub fn add_measurement(&mut self, metric_name: &str, timestamp: f64, value: f64) {
        let point = TimeSeriesPoint {
            timestamp,
            value,
            metadata: HashMap::new(),
        };

        self.time_series
            .entry(metric_name.to_string())
            .or_default()
            .push(point);
    }

    /// Add measurement with metadata
    pub fn add_measurement_with_metadata(
        &mut self,
        metric_name: &str,
        timestamp: f64,
        value: f64,
        metadata: HashMap<String, f64>,
    ) {
        let point = TimeSeriesPoint {
            timestamp,
            value,
            metadata,
        };

        self.time_series
            .entry(metric_name.to_string())
            .or_default()
            .push(point);
    }

    /// Detect concept drift in a metric time series
    pub fn detect_concept_drift(
        &mut self,
        metric_name: &str,
        method: DriftDetectionMethod,
    ) -> MetricsResult<ConceptDriftResult> {
        let series = self.time_series.get(metric_name).ok_or_else(|| {
            MetricsError::InvalidParameter(format!("Metric '{}' not found", metric_name))
        })?;

        if series.len() < self.config.min_samples {
            return Err(MetricsError::InvalidParameter(
                "Insufficient data for drift detection".to_string(),
            ));
        }

        match method {
            DriftDetectionMethod::KSTest => self.ks_test_drift_detection(series),
            DriftDetectionMethod::PageHinkley => self.page_hinkley_drift_detection(series),
            DriftDetectionMethod::ADWIN => self.adwin_drift_detection(series),
            DriftDetectionMethod::SPC => self.spc_drift_detection(series),
            DriftDetectionMethod::PSI => self.psi_drift_detection(series),
        }
    }

    /// Kolmogorov-Smirnov test for drift detection
    fn ks_test_drift_detection(
        &self,
        series: &[TimeSeriesPoint],
    ) -> MetricsResult<ConceptDriftResult> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let n = values.len();
        let split_point = n / 2;

        let sample1 = &values[..split_point];
        let sample2 = &values[split_point..];

        let (ks_statistic, p_value) = kolmogorov_smirnov_test(sample1, sample2);
        let drift_detected = p_value < self.config.significance_level;

        // Calculate drift magnitude as difference in means
        let mean1 = sample1.iter().sum::<f64>() / sample1.len() as f64;
        let mean2 = sample2.iter().sum::<f64>() / sample2.len() as f64;
        let drift_magnitude = (mean2 - mean1).abs();

        Ok(ConceptDriftResult {
            drift_detected,
            test_statistic: ks_statistic,
            p_value: Some(p_value),
            confidence: 1.0 - p_value,
            drift_point: if drift_detected {
                Some(series[split_point].timestamp)
            } else {
                None
            },
            drift_magnitude,
            method: DriftDetectionMethod::KSTest,
        })
    }

    /// Page-Hinkley test for drift detection
    fn page_hinkley_drift_detection(
        &self,
        series: &[TimeSeriesPoint],
    ) -> MetricsResult<ConceptDriftResult> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let threshold = 50.0; // Default threshold
        let alpha = 0.9999; // Forgetting factor

        let mut cumsum = 0.0;
        let mut mean = values[0];
        let mut drift_point = None;
        let mut max_cumsum = 0.0;

        for (i, &value) in values.iter().enumerate().skip(1) {
            mean = alpha * mean + (1.0 - alpha) * value;
            cumsum += value - mean;

            if cumsum > max_cumsum {
                max_cumsum = cumsum;
            }

            if max_cumsum - cumsum > threshold {
                drift_point = Some(series[i].timestamp);
                break;
            }
        }

        let drift_detected = drift_point.is_some();
        let drift_magnitude = if drift_detected {
            let split_idx = series
                .iter()
                .position(|p| Some(p.timestamp) == drift_point)
                .unwrap_or(series.len() / 2);
            let mean1 = values[..split_idx].iter().sum::<f64>() / split_idx as f64;
            let mean2 = values[split_idx..].iter().sum::<f64>() / (values.len() - split_idx) as f64;
            (mean2 - mean1).abs()
        } else {
            0.0
        };

        Ok(ConceptDriftResult {
            drift_detected,
            test_statistic: max_cumsum,
            p_value: None,
            confidence: if drift_detected { 0.95 } else { 0.05 },
            drift_point,
            drift_magnitude,
            method: DriftDetectionMethod::PageHinkley,
        })
    }

    /// ADWIN (Adaptive Windowing) drift detection
    fn adwin_drift_detection(
        &self,
        series: &[TimeSeriesPoint],
    ) -> MetricsResult<ConceptDriftResult> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let confidence = 0.95;
        let mut window = VecDeque::new();
        let mut drift_point = None;
        let mut drift_magnitude = 0.0;

        for (i, &value) in values.iter().enumerate() {
            window.push_back(value);

            // Check for drift using sliding window variance
            if window.len() >= self.config.window_size {
                let window_vec: Vec<f64> = window.iter().copied().collect();
                let variance = calculate_variance(&window_vec);
                let threshold = calculate_adwin_threshold(window.len(), confidence);

                if variance > threshold {
                    drift_point = Some(series[i].timestamp);
                    // Calculate magnitude as difference between window halves
                    let mid = window.len() / 2;
                    let first_half: Vec<f64> = window.iter().take(mid).copied().collect();
                    let second_half: Vec<f64> = window.iter().skip(mid).copied().collect();
                    let mean1 = first_half.iter().sum::<f64>() / first_half.len() as f64;
                    let mean2 = second_half.iter().sum::<f64>() / second_half.len() as f64;
                    drift_magnitude = (mean2 - mean1).abs();
                    break;
                }

                // Remove oldest element to maintain window size
                window.pop_front();
            }
        }

        Ok(ConceptDriftResult {
            drift_detected: drift_point.is_some(),
            test_statistic: drift_magnitude,
            p_value: None,
            confidence,
            drift_point,
            drift_magnitude,
            method: DriftDetectionMethod::ADWIN,
        })
    }

    /// Statistical Process Control (SPC) drift detection
    fn spc_drift_detection(&self, series: &[TimeSeriesPoint]) -> MetricsResult<ConceptDriftResult> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        // Calculate control limits
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std = calculate_std_dev(&values);
        let upper_limit = mean + 3.0 * std;
        let lower_limit = mean - 3.0 * std;

        // Check for points outside control limits
        let mut drift_point = None;
        let mut max_deviation = 0.0;

        for (i, &value) in values.iter().enumerate() {
            let deviation = if value > upper_limit {
                value - upper_limit
            } else if value < lower_limit {
                lower_limit - value
            } else {
                0.0
            };

            if deviation > max_deviation {
                max_deviation = deviation;
                if deviation > 0.0 {
                    drift_point = Some(series[i].timestamp);
                }
            }
        }

        Ok(ConceptDriftResult {
            drift_detected: drift_point.is_some(),
            test_statistic: max_deviation,
            p_value: None,
            confidence: if drift_point.is_some() { 0.997 } else { 0.003 }, // 3-sigma confidence
            drift_point,
            drift_magnitude: max_deviation,
            method: DriftDetectionMethod::SPC,
        })
    }

    /// Population Stability Index (PSI) drift detection
    fn psi_drift_detection(&self, series: &[TimeSeriesPoint]) -> MetricsResult<ConceptDriftResult> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let n = values.len();
        let split_point = n / 2;

        let baseline = &values[..split_point];
        let current = &values[split_point..];

        let psi = calculate_psi(baseline, current, 10)?;
        let threshold = 0.2; // Standard PSI threshold

        let drift_detected = psi > threshold;
        let drift_magnitude = if drift_detected {
            let mean1 = baseline.iter().sum::<f64>() / baseline.len() as f64;
            let mean2 = current.iter().sum::<f64>() / current.len() as f64;
            (mean2 - mean1).abs()
        } else {
            0.0
        };

        Ok(ConceptDriftResult {
            drift_detected,
            test_statistic: psi,
            p_value: None,
            confidence: if drift_detected { 0.95 } else { 0.05 },
            drift_point: if drift_detected {
                Some(series[split_point].timestamp)
            } else {
                None
            },
            drift_magnitude,
            method: DriftDetectionMethod::PSI,
        })
    }

    /// Analyze temporal trends in a metric
    pub fn analyze_temporal_trend(
        &self,
        metric_name: &str,
    ) -> MetricsResult<TemporalTrendAnalysis> {
        let series = self.time_series.get(metric_name).ok_or_else(|| {
            MetricsError::InvalidParameter(format!("Metric '{}' not found", metric_name))
        })?;

        if series.len() < self.config.min_samples {
            return Err(MetricsError::InvalidParameter(
                "Insufficient data for trend analysis".to_string(),
            ));
        }

        let timestamps: Vec<f64> = series.iter().map(|p| p.timestamp).collect();
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        // Linear regression for trend
        let (slope, intercept, r_squared) = linear_regression(&timestamps, &values);

        // Statistical significance of trend
        let n = values.len() as f64;
        let t_statistic = slope * ((n - 2.0) / (1.0 - r_squared)).sqrt();
        let p_value = 2.0 * (1.0 - t_distribution_cdf(t_statistic.abs(), n - 2.0));

        // Determine trend direction
        let trend_direction = if p_value < self.config.significance_level {
            if slope > 0.0 {
                TrendDirection::Increasing
            } else {
                TrendDirection::Decreasing
            }
        } else {
            let coefficient_variation =
                calculate_std_dev(&values) / values.iter().sum::<f64>() * values.len() as f64;
            if coefficient_variation > 0.3 {
                TrendDirection::Volatile
            } else {
                TrendDirection::Stable
            }
        };

        // Detect seasonal component
        let seasonal_component = detect_seasonality(&timestamps, &values);

        // Detect change points
        let change_points = detect_change_points(&timestamps, &values);

        Ok(TemporalTrendAnalysis {
            slope,
            intercept,
            r_squared,
            p_value,
            trend_direction,
            seasonal_component,
            change_points,
        })
    }

    /// Calculate adaptive weights based on temporal patterns
    pub fn calculate_adaptive_weights(&self, metric_name: &str) -> MetricsResult<Array1<f64>> {
        let series = self.time_series.get(metric_name).ok_or_else(|| {
            MetricsError::InvalidParameter(format!("Metric '{}' not found", metric_name))
        })?;

        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let n = values.len();

        // Exponential decay weights (more recent = higher weight)
        let decay_factor: f64 = 0.95;
        let mut weights = Vec::with_capacity(n);

        for i in 0..n {
            let weight = decay_factor.powi((n - 1 - i) as i32);
            weights.push(weight);
        }

        // Normalize weights
        let sum: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= sum;
        }

        Ok(Array1::from_vec(weights))
    }

    /// Get temporal stability measure
    pub fn temporal_stability(&self, metric_name: &str) -> MetricsResult<f64> {
        let series = self.time_series.get(metric_name).ok_or_else(|| {
            MetricsError::InvalidParameter(format!("Metric '{}' not found", metric_name))
        })?;

        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        // Calculate stability as inverse of coefficient of variation
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = calculate_std_dev(&values);
        let cv = std_dev / mean.abs();

        // Stability is 1 - normalized CV (bounded between 0 and 1)
        let stability = 1.0 / (1.0 + cv);

        Ok(stability)
    }

    /// Track metric evolution over time
    pub fn track_metric_evolution(&self, metric_name: &str) -> MetricsResult<MetricEvolution> {
        let series = self.time_series.get(metric_name).ok_or_else(|| {
            MetricsError::InvalidParameter(format!("Metric '{}' not found", metric_name))
        })?;

        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let timestamps: Vec<f64> = series.iter().map(|p| p.timestamp).collect();

        // Calculate rolling statistics
        let window_size = self.config.window_size.min(values.len() / 4);
        let rolling_means = calculate_rolling_mean(&values, window_size);
        let rolling_stds = calculate_rolling_std(&values, window_size);

        // Detect significant changes
        let significant_changes = detect_significant_changes(&values, &timestamps, 2.0);

        // Calculate overall trend
        let (slope, _, r_squared) = linear_regression(&timestamps, &values);

        Ok(MetricEvolution {
            timestamps: timestamps.clone(),
            values: values.clone(),
            rolling_means,
            rolling_stds,
            overall_trend_slope: slope,
            trend_strength: r_squared,
            significant_changes,
            evolution_score: calculate_evolution_score(&values),
        })
    }
}

/// Metric evolution tracking result
#[derive(Debug, Clone)]
pub struct MetricEvolution {
    pub timestamps: Vec<f64>,
    pub values: Vec<f64>,
    pub rolling_means: Vec<f64>,
    pub rolling_stds: Vec<f64>,
    pub overall_trend_slope: f64,
    pub trend_strength: f64,
    pub significant_changes: Vec<(f64, f64)>, // (timestamp, change_magnitude)
    pub evolution_score: f64,                 // 0-1 score indicating how much metric has evolved
}

/// Trait for drift detection algorithms
pub trait DriftDetector: Send + Sync {
    fn update(&mut self, value: f64) -> bool;
    fn reset(&mut self);
    fn is_change_detected(&self) -> bool;
}

// Helper functions

/// Kolmogorov-Smirnov test implementation
fn kolmogorov_smirnov_test(sample1: &[f64], sample2: &[f64]) -> (f64, f64) {
    let mut combined: Vec<f64> = sample1.iter().chain(sample2.iter()).copied().collect();
    combined.sort_by(|a, b| a.partial_cmp(b).unwrap());
    combined.dedup();

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let mut max_diff = 0.0;

    for &value in &combined {
        let cdf1 = sample1.iter().filter(|&&x| x <= value).count() as f64 / n1;
        let cdf2 = sample2.iter().filter(|&&x| x <= value).count() as f64 / n2;
        let diff = (cdf1 - cdf2).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // Approximate p-value calculation
    let en = (n1 * n2 / (n1 + n2)).sqrt();
    let lambda = max_diff * en;
    let p_value = 2.0 * (-2.0 * lambda * lambda).exp();

    (max_diff, p_value.min(1.0))
}

/// Calculate variance
fn calculate_variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Calculate standard deviation
fn calculate_std_dev(values: &[f64]) -> f64 {
    calculate_variance(values).sqrt()
}

/// Calculate ADWIN threshold
fn calculate_adwin_threshold(window_size: usize, confidence: f64) -> f64 {
    let n = window_size as f64;
    let delta = 1.0 - confidence;
    (2.0 * (2.0 / delta).ln() / n).sqrt()
}

/// Linear regression
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let sum_x = x.iter().sum::<f64>();
    let sum_y = y.iter().sum::<f64>();
    let sum_xy = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
    let sum_x2 = x.iter().map(|xi| xi * xi).sum::<f64>();
    let _sum_y2 = y.iter().map(|yi| yi * yi).sum::<f64>();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate R-squared
    let mean_y = sum_y / n;
    let ss_tot = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>();
    let ss_res = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
        .sum::<f64>();
    let r_squared = 1.0 - ss_res / ss_tot;

    (slope, intercept, r_squared)
}

/// Approximate t-distribution CDF
fn t_distribution_cdf(t: f64, df: f64) -> f64 {
    // Simplified approximation for large df
    if df > 30.0 {
        // Use normal approximation
        0.5 * (1.0 + erf(t / 2.0_f64.sqrt()))
    } else {
        // Very simple approximation
        0.5 + 0.5 * (t / (1.0 + t * t / df)).tanh()
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Calculate Population Stability Index
fn calculate_psi(baseline: &[f64], current: &[f64], bins: usize) -> MetricsResult<f64> {
    if baseline.is_empty() || current.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Find min and max for binning
    let min_val = baseline
        .iter()
        .chain(current.iter())
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = baseline
        .iter()
        .chain(current.iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if (max_val - min_val).abs() < f64::EPSILON {
        return Ok(0.0); // No variation
    }

    let bin_width = (max_val - min_val) / bins as f64;

    // Calculate expected and actual percentages
    let mut expected_counts = vec![0; bins];
    let mut actual_counts = vec![0; bins];

    for &value in baseline {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(bins - 1);
        expected_counts[bin_idx] += 1;
    }

    for &value in current {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(bins - 1);
        actual_counts[bin_idx] += 1;
    }

    // Calculate PSI
    let mut psi = 0.0;
    let baseline_total = baseline.len() as f64;
    let current_total = current.len() as f64;

    for i in 0..bins {
        let expected_pct = (expected_counts[i] as f64 / baseline_total).max(1e-10);
        let actual_pct = (actual_counts[i] as f64 / current_total).max(1e-10);

        psi += (actual_pct - expected_pct) * (actual_pct / expected_pct).ln();
    }

    Ok(psi)
}

/// Detect seasonality in time series
fn detect_seasonality(_timestamps: &[f64], values: &[f64]) -> Option<SeasonalComponent> {
    if values.len() < 12 {
        return None; // Need at least 12 points for seasonality detection
    }

    // Simple autocorrelation-based seasonality detection
    let max_lag = values.len() / 3;
    let mut best_period = 0;
    let mut best_correlation = 0.0;

    for lag in 2..max_lag {
        let correlation = autocorrelation(values, lag);
        if correlation > best_correlation {
            best_correlation = correlation;
            best_period = lag;
        }
    }

    if best_correlation > 0.3 {
        // Calculate amplitude and phase
        let period = best_period as f64;
        let amplitude = calculate_seasonal_amplitude(values, best_period);
        let phase = 0.0; // Simplified
        let strength = best_correlation;

        Some(SeasonalComponent {
            period,
            amplitude,
            phase,
            strength,
        })
    } else {
        None
    }
}

/// Calculate autocorrelation at given lag
fn autocorrelation(values: &[f64], lag: usize) -> f64 {
    if lag >= values.len() {
        return 0.0;
    }

    let n = values.len() - lag;
    let mean = values.iter().sum::<f64>() / values.len() as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let x = values[i] - mean;
        let y = values[i + lag] - mean;
        numerator += x * y;
        denominator += x * x;
    }

    if denominator > f64::EPSILON {
        numerator / denominator
    } else {
        0.0
    }
}

/// Calculate seasonal amplitude
fn calculate_seasonal_amplitude(values: &[f64], period: usize) -> f64 {
    if period == 0 || values.len() < period {
        return 0.0;
    }

    let _cycles = values.len() / period;
    let mut seasonal_values = vec![0.0; period];
    let mut counts = vec![0; period];

    for (i, &value) in values.iter().enumerate() {
        let season_idx = i % period;
        seasonal_values[season_idx] += value;
        counts[season_idx] += 1;
    }

    // Average by period
    for (i, count) in counts.iter().enumerate() {
        if *count > 0 {
            seasonal_values[i] /= *count as f64;
        }
    }

    // Calculate amplitude as max - min
    let max_seasonal = seasonal_values
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_seasonal = seasonal_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    max_seasonal - min_seasonal
}

/// Detect change points in time series
fn detect_change_points(timestamps: &[f64], values: &[f64]) -> Vec<f64> {
    let mut change_points = Vec::new();
    let min_segment_length = 10;

    if values.len() < 2 * min_segment_length {
        return change_points;
    }

    // Simple change point detection using running variance
    let window_size = values.len() / 4;

    for i in window_size..(values.len() - window_size) {
        let before = &values[(i - window_size)..i];
        let after = &values[i..(i + window_size)];

        let mean_before = before.iter().sum::<f64>() / before.len() as f64;
        let mean_after = after.iter().sum::<f64>() / after.len() as f64;

        let var_before = calculate_variance(before);
        let var_after = calculate_variance(after);

        // Detect significant change in mean or variance
        let mean_change = (mean_after - mean_before).abs();
        let var_change = (var_after - var_before).abs();

        let mean_threshold = calculate_std_dev(values) * 2.0;
        let var_threshold = calculate_variance(values) * 0.5;

        if mean_change > mean_threshold || var_change > var_threshold {
            change_points.push(timestamps[i]);
        }
    }

    change_points
}

/// Calculate rolling mean
fn calculate_rolling_mean(values: &[f64], window_size: usize) -> Vec<f64> {
    let mut rolling_means = Vec::new();

    for i in 0..values.len() {
        let start = if i >= window_size {
            i - window_size + 1
        } else {
            0
        };
        let end = i + 1;
        let window = &values[start..end];
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        rolling_means.push(mean);
    }

    rolling_means
}

/// Calculate rolling standard deviation
fn calculate_rolling_std(values: &[f64], window_size: usize) -> Vec<f64> {
    let mut rolling_stds = Vec::new();

    for i in 0..values.len() {
        let start = if i >= window_size {
            i - window_size + 1
        } else {
            0
        };
        let end = i + 1;
        let window = &values[start..end];
        let std = calculate_std_dev(window);
        rolling_stds.push(std);
    }

    rolling_stds
}

/// Detect significant changes
fn detect_significant_changes(
    values: &[f64],
    timestamps: &[f64],
    threshold_multiplier: f64,
) -> Vec<(f64, f64)> {
    let mut changes = Vec::new();
    let overall_std = calculate_std_dev(values);
    let threshold = overall_std * threshold_multiplier;

    for i in 1..values.len() {
        let change = (values[i] - values[i - 1]).abs();
        if change > threshold {
            changes.push((timestamps[i], change));
        }
    }

    changes
}

/// Calculate evolution score
fn calculate_evolution_score(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    // Calculate relative change from first to last value
    let first = values[0];
    let last = values[values.len() - 1];
    let total_change = if first.abs() > f64::EPSILON {
        (last - first).abs() / first.abs()
    } else {
        last.abs()
    };

    // Normalize to 0-1 scale (sigmoid transformation)
    1.0 / (1.0 + (-total_change).exp())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_temporal_analyzer_creation() {
        let config = WindowConfig::default();
        let analyzer = TemporalMetricsAnalyzer::new(config);
        assert_eq!(analyzer.time_series.len(), 0);
    }

    #[test]
    fn test_add_measurements() {
        let mut analyzer = TemporalMetricsAnalyzer::new(WindowConfig::default());
        analyzer.add_measurement("accuracy", 1.0, 0.95);
        analyzer.add_measurement("accuracy", 2.0, 0.94);
        analyzer.add_measurement("accuracy", 3.0, 0.93);

        let series = analyzer.time_series.get("accuracy").unwrap();
        assert_eq!(series.len(), 3);
        assert_eq!(series[0].value, 0.95);
        assert_eq!(series[1].value, 0.94);
        assert_eq!(series[2].value, 0.93);
    }

    #[test]
    fn test_ks_drift_detection() {
        let mut analyzer = TemporalMetricsAnalyzer::new(WindowConfig::default());

        // Add stable data first
        for i in 0..50 {
            analyzer.add_measurement("metric", i as f64, 0.9 + 0.01 * (i as f64 % 10.0) / 10.0);
        }

        // Add drifted data
        for i in 50..100 {
            analyzer.add_measurement("metric", i as f64, 0.7 + 0.01 * (i as f64 % 10.0) / 10.0);
        }

        let result = analyzer
            .detect_concept_drift("metric", DriftDetectionMethod::KSTest)
            .unwrap();
        assert!(result.drift_detected);
        assert!(result.drift_magnitude > 0.1);
    }

    #[test]
    fn test_temporal_trend_analysis() {
        let mut analyzer = TemporalMetricsAnalyzer::new(WindowConfig::default());

        // Add trending data
        for i in 0..50 {
            let value = 0.5 + 0.01 * i as f64; // Clear upward trend
            analyzer.add_measurement("metric", i as f64, value);
        }

        let trend = analyzer.analyze_temporal_trend("metric").unwrap();
        assert!(trend.slope > 0.0);
        assert_eq!(trend.trend_direction, TrendDirection::Increasing);
        assert!(trend.r_squared > 0.9); // Strong linear trend
    }

    #[test]
    fn test_adaptive_weights() {
        let mut analyzer = TemporalMetricsAnalyzer::new(WindowConfig::default());

        for i in 0..10 {
            analyzer.add_measurement("metric", i as f64, i as f64);
        }

        let weights = analyzer.calculate_adaptive_weights("metric").unwrap();
        assert_eq!(weights.len(), 10);

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        // Later weights should be larger (exponential decay)
        assert!(weights[9] > weights[0]);
    }

    #[test]
    fn test_temporal_stability() {
        let mut analyzer = TemporalMetricsAnalyzer::new(WindowConfig::default());

        // Add stable data
        for i in 0..20 {
            analyzer.add_measurement("stable", i as f64, 0.95);
        }

        // Add volatile data
        for i in 0..20 {
            let value = 0.5 + 0.5 * ((i as f64 * 0.5).sin());
            analyzer.add_measurement("volatile", i as f64, value);
        }

        let stable_score = analyzer.temporal_stability("stable").unwrap();
        let volatile_score = analyzer.temporal_stability("volatile").unwrap();

        assert!(stable_score > volatile_score);
        assert!(stable_score > 0.9); // Should be very stable
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let (slope, intercept, r_squared) = linear_regression(&x, &y);

        assert_abs_diff_eq!(slope, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(intercept, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r_squared, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kolmogorov_smirnov() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Same distribution

        let (ks_stat, p_value) = kolmogorov_smirnov_test(&sample1, &sample2);

        assert_abs_diff_eq!(ks_stat, 0.0, epsilon = 1e-10);
        assert!(p_value > 0.5); // Should not be significant
    }

    #[test]
    fn test_variance_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = calculate_variance(&values);
        let expected_variance = 2.0; // Variance of 1,2,3,4,5

        assert_abs_diff_eq!(variance, expected_variance, epsilon = 1e-10);
    }

    #[test]
    fn test_psi_calculation() {
        let baseline = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let current = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Same distribution

        let psi = calculate_psi(&baseline, &current, 5).unwrap();
        assert!(psi < 0.1); // Should be low for same distribution
    }

    #[test]
    fn test_seasonality_detection() {
        // Create data with seasonal pattern (period = 12)
        let timestamps: Vec<f64> = (0..36).map(|i| i as f64).collect(); // Use more data points
        let values: Vec<f64> = (0..36)
            .map(|i| (i as f64 * std::f64::consts::PI / 6.0).sin())
            .collect();

        let seasonal = detect_seasonality(&timestamps, &values);
        assert!(seasonal.is_some());

        let seasonal = seasonal.unwrap();
        assert!(seasonal.period > 10.0 && seasonal.period < 15.0); // Should detect ~12 period
        assert!(seasonal.strength > 0.3);
    }

    #[test]
    fn test_rolling_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolling_means = calculate_rolling_mean(&values, 3);

        assert_eq!(rolling_means.len(), 5);
        assert_eq!(rolling_means[0], 1.0); // First point
        assert_eq!(rolling_means[1], 1.5); // Mean of [1,2]
        assert_eq!(rolling_means[2], 2.0); // Mean of [1,2,3]
        assert_eq!(rolling_means[3], 3.0); // Mean of [2,3,4]
        assert_eq!(rolling_means[4], 4.0); // Mean of [3,4,5]
    }
}
