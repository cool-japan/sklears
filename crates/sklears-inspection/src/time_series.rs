//! Time Series Interpretability Module
//!
//! This module provides specialized interpretability methods for time series models,
//! including temporal importance analysis, seasonal decomposition explanations,
//! trend analysis, lag importance, and dynamic time warping explanations.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for time series interpretability methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfig {
    /// Maximum lag to consider for lag importance analysis
    pub max_lag: usize,

    /// Window size for trend analysis
    pub trend_window: usize,

    /// Number of seasonal periods to consider
    pub seasonal_periods: Vec<usize>,

    /// Alpha for exponential smoothing
    pub alpha: Float,

    /// Beta for trend smoothing
    pub beta: Float,

    /// Gamma for seasonal smoothing
    pub gamma: Float,

    /// DTW window constraint (percentage of series length)
    pub dtw_window: Float,

    /// Method for seasonal decomposition
    pub decomposition_method: DecompositionMethod,

    /// Number of bootstrap samples for significance testing
    pub n_bootstrap: usize,

    /// Significance level for statistical tests
    pub significance_level: Float,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            max_lag: 20,
            trend_window: 5,
            seasonal_periods: vec![7, 12, 24, 52], // Daily, monthly, hourly, weekly
            alpha: 0.3,
            beta: 0.3,
            gamma: 0.3,
            dtw_window: 0.1,
            decomposition_method: DecompositionMethod::STL,
            n_bootstrap: 1000,
            significance_level: 0.05,
        }
    }
}

/// Methods for seasonal decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionMethod {
    /// Seasonal and Trend decomposition using Loess
    STL,
    /// Classical additive decomposition
    Additive,
    /// Classical multiplicative decomposition
    Multiplicative,
    /// X-13ARIMA-SEATS (simplified)
    X13,
}

/// Result of temporal importance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalImportance {
    /// Importance scores for each time step
    pub temporal_scores: Array1<Float>,

    /// Window-based importance scores
    pub window_scores: HashMap<usize, Array1<Float>>,

    /// Statistical significance of temporal patterns
    pub significance_map: Array1<bool>,

    /// Confidence intervals for importance scores
    pub confidence_intervals: Array2<Float>, // [lower, upper] for each time step

    /// Metadata about the analysis
    pub metadata: HashMap<String, String>,
}

/// Result of seasonal decomposition explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalDecomposition {
    /// Original time series
    pub original: Array1<Float>,

    /// Trend component
    pub trend: Array1<Float>,

    /// Seasonal component
    pub seasonal: Array1<Float>,

    /// Residual component
    pub residual: Array1<Float>,

    /// Strength of trend (0-1)
    pub trend_strength: Float,

    /// Strength of seasonality (0-1)
    pub seasonal_strength: Float,

    /// Detected seasonal periods
    pub seasonal_periods: Vec<usize>,

    /// Explained variance by each component
    pub explained_variance: HashMap<String, Float>,
}

/// Result of trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Overall trend direction (positive, negative, stationary)
    pub trend_direction: TrendDirection,

    /// Trend magnitude
    pub trend_magnitude: Float,

    /// Local trend segments
    pub trend_segments: Vec<TrendSegment>,

    /// Change points in the trend
    pub change_points: Vec<usize>,

    /// Trend significance test results
    pub significance_tests: HashMap<String, Float>,

    /// Trend forecasting quality metrics
    pub forecast_metrics: HashMap<String, Float>,
}

/// Direction of trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing
    Increasing,
    /// Decreasing
    Decreasing,
    /// Stationary
    Stationary,
    /// Cyclical
    Cyclical,
}

/// Individual trend segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSegment {
    /// Start index of segment
    pub start: usize,

    /// End index of segment
    pub end: usize,

    /// Slope of segment
    pub slope: Float,

    /// R-squared for segment
    pub r_squared: Float,

    /// Confidence in segment
    pub confidence: Float,
}

/// Result of lag importance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagImportance {
    /// Importance scores for each lag
    pub lag_scores: Array1<Float>,

    /// Cross-correlation with target at each lag
    pub cross_correlations: Array1<Float>,

    /// Partial autocorrelation function values
    pub pacf_values: Array1<Float>,

    /// Granger causality test results
    pub granger_causality: Array1<Float>,

    /// Optimal lag order
    pub optimal_lag: usize,

    /// Information criteria for different lag orders
    pub information_criteria: HashMap<String, Array1<Float>>,
}

/// Result of dynamic time warping explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DTWExplanation {
    /// DTW distance
    pub distance: Float,

    /// Optimal warping path
    pub warping_path: Array2<usize>,

    /// Local distance matrix
    pub distance_matrix: Array2<Float>,

    /// Alignment explanation
    pub alignment_explanation: Vec<AlignmentSegment>,

    /// Feature importance in DTW space
    pub dtw_feature_importance: Array1<Float>,

    /// Normalized warping path
    pub normalized_path: Array2<Float>,
}

/// Alignment segment for DTW explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentSegment {
    /// Query indices involved in alignment
    pub query_range: (usize, usize),

    /// Reference indices involved in alignment
    pub reference_range: (usize, usize),

    /// Cost of this alignment segment
    pub alignment_cost: Float,

    /// Type of alignment (match, expansion, compression)
    pub alignment_type: AlignmentType,
}

/// Type of DTW alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentType {
    /// Match
    Match, // 1:1 alignment
    /// Expansion
    Expansion, // 1:many alignment (query expanded)
    /// Compression
    Compression, // many:1 alignment (query compressed)
}

/// Main time series interpretability analyzer
#[derive(Debug, Clone)]
pub struct TimeSeriesAnalyzer {
    config: TimeSeriesConfig,
}

impl TimeSeriesAnalyzer {
    /// Create new time series analyzer
    pub fn new(config: TimeSeriesConfig) -> Self {
        Self { config }
    }

    /// Create analyzer with default configuration
    pub fn default() -> Self {
        Self::new(TimeSeriesConfig::default())
    }

    /// Analyze temporal importance of features
    pub fn temporal_importance(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<Float>,
        model_fn: &dyn Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
    ) -> SklResult<TemporalImportance> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Compute baseline predictions
        let baseline_predictions = model_fn(X)?;
        let baseline_score = self.compute_model_score(y, &baseline_predictions.view())?;

        // Analyze importance for each time step
        let mut temporal_scores = Array1::zeros(n_samples);
        let mut significance_map = Array1::from_elem(n_samples, false);
        let mut confidence_intervals = Array2::zeros((n_samples, 2));

        for t in 0..n_samples {
            // Permute features at time t
            let mut X_perturbed = X.to_owned();
            self.permute_time_step(&mut X_perturbed, t)?;

            // Compute perturbed predictions
            let perturbed_predictions = model_fn(&X_perturbed.view())?;
            let perturbed_score = self.compute_model_score(y, &perturbed_predictions.view())?;

            // Importance as difference in performance
            let importance = baseline_score - perturbed_score;
            temporal_scores[t] = importance;

            // Bootstrap for confidence intervals
            let bootstrap_scores =
                self.bootstrap_temporal_importance(X, y, model_fn, t, baseline_score)?;

            let (lower, upper) = self.compute_confidence_interval(&bootstrap_scores.view())?;
            confidence_intervals[[t, 0]] = lower;
            confidence_intervals[[t, 1]] = upper;

            // Significance test
            significance_map[t] = self.test_significance(&bootstrap_scores.view(), 0.0)?;
        }

        // Compute window-based importance
        let window_scores = self.compute_window_importance(X, y, model_fn)?;

        let mut metadata = HashMap::new();
        metadata.insert("method".to_string(), "permutation_importance".to_string());
        metadata.insert("n_samples".to_string(), n_samples.to_string());
        metadata.insert("baseline_score".to_string(), baseline_score.to_string());

        Ok(TemporalImportance {
            temporal_scores,
            window_scores,
            significance_map,
            confidence_intervals,
            metadata,
        })
    }

    /// Perform seasonal decomposition
    pub fn seasonal_decomposition(
        &self,
        time_series: &ArrayView1<Float>,
    ) -> SklResult<SeasonalDecomposition> {
        let n = time_series.len();

        match self.config.decomposition_method {
            DecompositionMethod::STL => self.stl_decomposition(time_series),
            DecompositionMethod::Additive => self.additive_decomposition(time_series),
            DecompositionMethod::Multiplicative => self.multiplicative_decomposition(time_series),
            DecompositionMethod::X13 => self.x13_decomposition(time_series),
        }
    }

    /// Analyze trends in time series
    pub fn trend_analysis(&self, time_series: &ArrayView1<Float>) -> SklResult<TrendAnalysis> {
        let n = time_series.len();

        // Compute overall trend using linear regression
        let x: Array1<Float> = (0..n).map(|i| i as Float).collect();
        let (slope, _intercept, r_squared) = self.linear_regression(&x.view(), time_series)?;

        // Determine trend direction
        let trend_direction = if slope.abs() < 1e-6 {
            TrendDirection::Stationary
        } else if slope > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        };

        let trend_magnitude = slope.abs();

        // Detect change points
        let change_points = self.detect_change_points(time_series)?;

        // Create trend segments
        let trend_segments = self.create_trend_segments(time_series, &change_points)?;

        // Statistical significance tests
        let mut significance_tests = HashMap::new();
        significance_tests.insert(
            "mann_kendall".to_string(),
            self.mann_kendall_test(time_series)?,
        );
        significance_tests.insert("adf_test".to_string(), self.adf_test(time_series)?);

        // Forecast metrics
        let forecast_metrics = self.compute_forecast_metrics(time_series)?;

        Ok(TrendAnalysis {
            trend_direction,
            trend_magnitude,
            trend_segments,
            change_points,
            significance_tests,
            forecast_metrics,
        })
    }

    /// Analyze lag importance
    pub fn lag_importance(
        &self,
        time_series: &ArrayView1<Float>,
        target: &ArrayView1<Float>,
    ) -> SklResult<LagImportance> {
        let max_lag = self.config.max_lag.min(time_series.len() / 4);

        // Compute cross-correlations
        let cross_correlations = self.compute_cross_correlation(time_series, target, max_lag)?;

        // Compute partial autocorrelation function
        let pacf_values = self.compute_pacf(time_series, max_lag)?;

        // Granger causality tests
        let granger_causality = self.granger_causality_test(time_series, target, max_lag)?;

        // Lag importance as combination of metrics
        let mut lag_scores = Array1::zeros(max_lag + 1);
        for lag in 0..=max_lag {
            lag_scores[lag] =
                (cross_correlations[lag].abs() + pacf_values[lag].abs() + granger_causality[lag])
                    / 3.0;
        }

        // Find optimal lag
        let optimal_lag = lag_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Information criteria
        let information_criteria =
            self.compute_information_criteria(time_series, target, max_lag)?;

        Ok(LagImportance {
            lag_scores,
            cross_correlations,
            pacf_values,
            granger_causality,
            optimal_lag,
            information_criteria,
        })
    }

    /// Generate DTW explanations
    pub fn dtw_explanation(
        &self,
        query: &ArrayView1<Float>,
        reference: &ArrayView1<Float>,
    ) -> SklResult<DTWExplanation> {
        // Compute DTW distance and path
        let (distance, warping_path, distance_matrix) = self.compute_dtw(query, reference)?;

        // Create alignment explanation
        let alignment_explanation =
            self.create_alignment_explanation(&warping_path, &distance_matrix)?;

        // Compute feature importance in DTW space
        let dtw_feature_importance =
            self.compute_dtw_feature_importance(query, reference, &warping_path)?;

        // Normalize warping path
        let normalized_path =
            self.normalize_warping_path(&warping_path, query.len(), reference.len())?;

        Ok(DTWExplanation {
            distance,
            warping_path,
            distance_matrix,
            alignment_explanation,
            dtw_feature_importance,
            normalized_path,
        })
    }

    // Private helper methods

    fn compute_model_score(
        &self,
        y_true: &ArrayView1<Float>,
        y_pred: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        // Use R-squared score
        let y_mean = y_true.mean().unwrap();
        let ss_tot: Float = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: Float = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| (yt - yp).powi(2))
            .sum();

        if ss_tot == 0.0 {
            Ok(1.0)
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }

    fn permute_time_step(&self, X: &mut Array2<Float>, t: usize) -> SklResult<()> {
        let n_features = X.ncols();

        // Permute all features at time step t
        for j in 0..n_features {
            // Find a random other time step to swap with
            let other_t = if X.nrows() > 1 {
                if t == 0 {
                    1
                } else {
                    0
                }
            } else {
                return Ok(());
            };

            let temp = X[[t, j]];
            X[[t, j]] = X[[other_t, j]];
            X[[other_t, j]] = temp;
        }

        Ok(())
    }

    fn bootstrap_temporal_importance(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<Float>,
        model_fn: &dyn Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
        time_step: usize,
        baseline_score: Float,
    ) -> SklResult<Array1<Float>> {
        let mut bootstrap_scores = Array1::zeros(self.config.n_bootstrap);

        for i in 0..self.config.n_bootstrap {
            // Create bootstrap sample
            let mut X_boot = X.to_owned();
            self.permute_time_step(&mut X_boot, time_step)?;

            // Compute score
            let predictions = model_fn(&X_boot.view())?;
            let score = self.compute_model_score(y, &predictions.view())?;
            bootstrap_scores[i] = baseline_score - score;
        }

        Ok(bootstrap_scores)
    }

    fn compute_confidence_interval(&self, scores: &ArrayView1<Float>) -> SklResult<(Float, Float)> {
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha = self.config.significance_level;
        let lower_idx = ((alpha / 2.0) * sorted_scores.len() as Float) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * sorted_scores.len() as Float) as usize;

        let lower = sorted_scores.get(lower_idx).copied().unwrap_or(0.0);
        let upper = sorted_scores
            .get(upper_idx.min(sorted_scores.len() - 1))
            .copied()
            .unwrap_or(0.0);

        Ok((lower, upper))
    }

    fn test_significance(&self, scores: &ArrayView1<Float>, null_value: Float) -> SklResult<bool> {
        // Simple t-test against null hypothesis
        let mean = scores.mean().unwrap();
        let std = scores.std(0.0);
        let n = scores.len() as Float;

        if std == 0.0 {
            return Ok(false);
        }

        let t_stat = (mean - null_value) / (std / n.sqrt());
        let p_value = 2.0 * (1.0 - self.student_t_cdf(t_stat.abs(), n - 1.0));

        Ok(p_value < self.config.significance_level)
    }

    fn student_t_cdf(&self, t: Float, df: Float) -> Float {
        // Simplified Student's t CDF approximation
        // For production use, should use proper statistical library
        let x = t / (df + t * t).sqrt();
        0.5 + x * (0.5 + x * x * (-0.125 + x * x * 0.0625))
    }

    fn compute_window_importance(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<Float>,
        model_fn: &dyn Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
    ) -> SklResult<HashMap<usize, Array1<Float>>> {
        let mut window_scores = HashMap::new();
        let n_samples = X.nrows();

        for &window_size in &[3, 5, 7, 10] {
            if window_size >= n_samples {
                continue;
            }

            let mut scores = Array1::zeros(n_samples - window_size + 1);

            for start in 0..=(n_samples - window_size) {
                // Permute entire window
                let mut X_windowed = X.to_owned();
                for t in start..(start + window_size) {
                    self.permute_time_step(&mut X_windowed, t)?;
                }

                let baseline_predictions = model_fn(X)?;
                let perturbed_predictions = model_fn(&X_windowed.view())?;

                let baseline_score = self.compute_model_score(y, &baseline_predictions.view())?;
                let perturbed_score = self.compute_model_score(y, &perturbed_predictions.view())?;

                scores[start] = baseline_score - perturbed_score;
            }

            window_scores.insert(window_size, scores);
        }

        Ok(window_scores)
    }

    fn stl_decomposition(
        &self,
        time_series: &ArrayView1<Float>,
    ) -> SklResult<SeasonalDecomposition> {
        // Simplified STL decomposition
        let n = time_series.len();
        let seasonal_period = self.config.seasonal_periods.first().copied().unwrap_or(12);

        // Trend component using moving average
        let trend = self.compute_moving_average(time_series, self.config.trend_window)?;

        // Detrended series
        let detrended: Array1<Float> = time_series
            .iter()
            .zip(trend.iter())
            .map(|(&x, &t)| x - t)
            .collect();

        // Seasonal component
        let seasonal = self.extract_seasonal_component(&detrended.view(), seasonal_period)?;

        // Residual
        let residual: Array1<Float> = time_series
            .iter()
            .zip(trend.iter().zip(seasonal.iter()))
            .map(|(&x, (&t, &s))| x - t - s)
            .collect();

        // Compute strengths
        let trend_strength = self.compute_component_strength(&trend.view(), time_series)?;
        let seasonal_strength = self.compute_component_strength(&seasonal.view(), time_series)?;

        let mut explained_variance = HashMap::new();
        explained_variance.insert("trend".to_string(), trend_strength);
        explained_variance.insert("seasonal".to_string(), seasonal_strength);
        explained_variance.insert(
            "residual".to_string(),
            1.0 - trend_strength - seasonal_strength,
        );

        Ok(SeasonalDecomposition {
            original: time_series.to_owned(),
            trend,
            seasonal,
            residual,
            trend_strength,
            seasonal_strength,
            seasonal_periods: vec![seasonal_period],
            explained_variance,
        })
    }

    fn additive_decomposition(
        &self,
        time_series: &ArrayView1<Float>,
    ) -> SklResult<SeasonalDecomposition> {
        // Simplified additive decomposition
        self.stl_decomposition(time_series)
    }

    fn multiplicative_decomposition(
        &self,
        time_series: &ArrayView1<Float>,
    ) -> SklResult<SeasonalDecomposition> {
        // Take log, perform additive decomposition, then exp
        let log_series: Array1<Float> = time_series.iter().map(|&x| (x.max(1e-10)).ln()).collect();

        let log_decomp = self.stl_decomposition(&log_series.view())?;

        Ok(SeasonalDecomposition {
            original: time_series.to_owned(),
            trend: log_decomp.trend.mapv(|x| x.exp()),
            seasonal: log_decomp.seasonal.mapv(|x| x.exp()),
            residual: log_decomp.residual.mapv(|x| x.exp()),
            trend_strength: log_decomp.trend_strength,
            seasonal_strength: log_decomp.seasonal_strength,
            seasonal_periods: log_decomp.seasonal_periods,
            explained_variance: log_decomp.explained_variance,
        })
    }

    fn x13_decomposition(
        &self,
        time_series: &ArrayView1<Float>,
    ) -> SklResult<SeasonalDecomposition> {
        // Simplified X-13 method (just use STL for now)
        self.stl_decomposition(time_series)
    }

    fn compute_moving_average(
        &self,
        series: &ArrayView1<Float>,
        window: usize,
    ) -> SklResult<Array1<Float>> {
        let n = series.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(n);

            let sum: Float = series.slice(scirs2_core::ndarray::s![start..end]).sum();
            result[i] = sum / (end - start) as Float;
        }

        Ok(result)
    }

    fn extract_seasonal_component(
        &self,
        detrended: &ArrayView1<Float>,
        period: usize,
    ) -> SklResult<Array1<Float>> {
        let n = detrended.len();
        let mut seasonal = Array1::zeros(n);

        if period == 0 {
            return Ok(seasonal);
        }

        // Compute average for each seasonal index
        let mut seasonal_means = vec![0.0; period];
        let mut counts = vec![0; period];

        for (i, &value) in detrended.iter().enumerate() {
            let seasonal_idx = i % period;
            seasonal_means[seasonal_idx] += value;
            counts[seasonal_idx] += 1;
        }

        for i in 0..period {
            if counts[i] > 0 {
                seasonal_means[i] /= counts[i] as Float;
            }
        }

        // Fill seasonal component
        for i in 0..n {
            seasonal[i] = seasonal_means[i % period];
        }

        Ok(seasonal)
    }

    fn compute_component_strength(
        &self,
        component: &ArrayView1<Float>,
        original: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        let component_var = component.var(0.0);
        let original_var = original.var(0.0);

        if original_var == 0.0 {
            Ok(0.0)
        } else {
            Ok((component_var / original_var).min(1.0))
        }
    }

    fn linear_regression(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<(Float, Float, Float)> {
        let n = x.len() as Float;
        let x_mean = x.mean().unwrap();
        let y_mean = y.mean().unwrap();

        let numerator: Float = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let denominator: Float = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();

        if denominator == 0.0 {
            return Ok((0.0, y_mean, 0.0));
        }

        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;

        // Compute R-squared
        let ss_tot: Float = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: Float = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| {
                let predicted = slope * xi + intercept;
                (yi - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

        Ok((slope, intercept, r_squared))
    }

    fn detect_change_points(&self, time_series: &ArrayView1<Float>) -> SklResult<Vec<usize>> {
        // Simple change point detection using moving variance
        let window = self.config.trend_window;
        let mut change_points = Vec::new();

        if time_series.len() < 2 * window {
            return Ok(change_points);
        }

        let mut variances = Vec::new();
        for i in window..(time_series.len() - window) {
            let window_data =
                time_series.slice(scirs2_core::ndarray::s![(i - window)..(i + window)]);
            variances.push(window_data.var(0.0));
        }

        // Find significant changes in variance
        for i in 1..variances.len() {
            let ratio = variances[i] / variances[i - 1].max(1e-10);
            if !(0.5..=2.0).contains(&ratio) {
                change_points.push(i + window);
            }
        }

        Ok(change_points)
    }

    fn create_trend_segments(
        &self,
        time_series: &ArrayView1<Float>,
        change_points: &[usize],
    ) -> SklResult<Vec<TrendSegment>> {
        let mut segments = Vec::new();
        let mut points = vec![0];
        points.extend_from_slice(change_points);
        points.push(time_series.len());

        for i in 0..(points.len() - 1) {
            let start = points[i];
            let end = points[i + 1];

            if end - start < 2 {
                continue;
            }

            let x: Array1<Float> = (start..end).map(|j| j as Float).collect();
            let y = time_series.slice(scirs2_core::ndarray::s![start..end]);

            let (slope, _intercept, r_squared) = self.linear_regression(&x.view(), &y)?;
            let confidence = r_squared; // Simplified confidence measure

            segments.push(TrendSegment {
                start,
                end,
                slope,
                r_squared,
                confidence,
            });
        }

        Ok(segments)
    }

    fn mann_kendall_test(&self, time_series: &ArrayView1<Float>) -> SklResult<Float> {
        // Simplified Mann-Kendall test for trend
        let n = time_series.len();
        let mut s = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                if time_series[j] > time_series[i] {
                    s += 1;
                } else if time_series[j] < time_series[i] {
                    s -= 1;
                }
            }
        }

        let var_s = (n * (n - 1) * (2 * n + 5)) as Float / 18.0;
        let z = if s == 0 {
            0.0
        } else if s > 0 {
            (s as Float - 1.0) / var_s.sqrt()
        } else {
            (s as Float + 1.0) / var_s.sqrt()
        };

        // Return approximate p-value
        Ok(2.0 * (1.0 - self.standard_normal_cdf(z.abs())))
    }

    fn standard_normal_cdf(&self, z: Float) -> Float {
        // Approximation for standard normal CDF
        0.5 * (1.0 + ((z / 2.0_f64.sqrt()).tanh() as Float))
    }

    fn adf_test(&self, time_series: &ArrayView1<Float>) -> SklResult<Float> {
        // Simplified Augmented Dickey-Fuller test
        // In practice, would use proper statistical library
        let n = time_series.len();
        if n < 3 {
            return Ok(1.0); // Cannot reject stationarity
        }

        // Compute first differences
        let mut diffs = Vec::new();
        for i in 1..n {
            diffs.push(time_series[i] - time_series[i - 1]);
        }

        // Simple test statistic based on variance of differences
        let diff_var = diffs.iter().map(|&x| x * x).sum::<Float>() / diffs.len() as Float;
        let series_var = time_series.var(0.0);

        // Approximate p-value
        Ok((diff_var / series_var.max(1e-10)).min(1.0))
    }

    fn compute_forecast_metrics(
        &self,
        time_series: &ArrayView1<Float>,
    ) -> SklResult<HashMap<String, Float>> {
        let mut metrics = HashMap::new();

        // Simple metrics based on autocorrelation
        let acf_lag1 = self.compute_autocorrelation(time_series, 1)?;
        metrics.insert("predictability".to_string(), acf_lag1.abs());

        // Variance-based metrics
        let variance = time_series.var(0.0);
        metrics.insert("volatility".to_string(), variance.sqrt());

        Ok(metrics)
    }

    fn compute_autocorrelation(&self, series: &ArrayView1<Float>, lag: usize) -> SklResult<Float> {
        let n = series.len();
        if lag >= n {
            return Ok(0.0);
        }

        let mean = series.mean().unwrap();

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..(n - lag) {
            numerator += (series[i] - mean) * (series[i + lag] - mean);
        }

        for i in 0..n {
            denominator += (series[i] - mean).powi(2);
        }

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn compute_cross_correlation(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        max_lag: usize,
    ) -> SklResult<Array1<Float>> {
        let mut correlations = Array1::zeros(max_lag + 1);

        for lag in 0..=max_lag {
            correlations[lag] = self.cross_correlation_at_lag(x, y, lag)?;
        }

        Ok(correlations)
    }

    fn cross_correlation_at_lag(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        lag: usize,
    ) -> SklResult<Float> {
        let n = x.len().min(y.len());
        if lag >= n {
            return Ok(0.0);
        }

        let effective_n = n - lag;
        let x_slice = x.slice(scirs2_core::ndarray::s![lag..]);
        let y_slice = y.slice(scirs2_core::ndarray::s![..effective_n]);

        let x_mean = x_slice.mean().unwrap();
        let y_mean = y_slice.mean().unwrap();

        let mut numerator = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for i in 0..effective_n {
            let x_dev = x_slice[i] - x_mean;
            let y_dev = y_slice[i] - y_mean;

            numerator += x_dev * y_dev;
            x_var += x_dev * x_dev;
            y_var += y_dev * y_dev;
        }

        let denominator = (x_var * y_var).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn compute_pacf(&self, series: &ArrayView1<Float>, max_lag: usize) -> SklResult<Array1<Float>> {
        // Simplified PACF computation using Yule-Walker equations
        let mut pacf_values = Array1::zeros(max_lag + 1);
        pacf_values[0] = 1.0; // PACF at lag 0 is always 1

        // Compute autocorrelations first
        let mut acf = Array1::zeros(max_lag + 1);
        for lag in 0..=max_lag {
            acf[lag] = self.compute_autocorrelation(series, lag)?;
        }

        // Compute PACF using Durbin-Levinson algorithm (simplified)
        for k in 1..=max_lag {
            if k == 1 {
                pacf_values[k] = acf[1];
            } else {
                // Simplified calculation
                pacf_values[k] = acf[k] * (1.0 - pacf_values[k - 1].powi(2));
            }
        }

        Ok(pacf_values)
    }

    fn granger_causality_test(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        max_lag: usize,
    ) -> SklResult<Array1<Float>> {
        // Simplified Granger causality test
        let mut causality_values = Array1::zeros(max_lag + 1);

        for lag in 1..=max_lag {
            // Compare prediction errors with and without x
            let error_with_x = self.compute_prediction_error(y, Some(x), lag)?;
            let error_without_x = self.compute_prediction_error(y, None, lag)?;

            // F-statistic approximation
            let improvement = (error_without_x - error_with_x) / error_without_x.max(1e-10);
            causality_values[lag] = improvement.max(0.0);
        }

        Ok(causality_values)
    }

    fn compute_prediction_error(
        &self,
        y: &ArrayView1<Float>,
        x: Option<&ArrayView1<Float>>,
        lag: usize,
    ) -> SklResult<Float> {
        let n = y.len();
        if lag >= n {
            return Ok(1.0);
        }

        let mut error = 0.0;
        let mut count = 0;

        for i in lag..n {
            // Simple AR model
            let mut prediction = 0.0;
            for j in 1..=lag {
                if i >= j {
                    prediction += y[i - j] * (1.0 / lag as Float);

                    // Add x terms if provided
                    if let Some(x_series) = x {
                        if i >= j && i - j < x_series.len() {
                            prediction += x_series[i - j] * (0.5 / lag as Float);
                        }
                    }
                }
            }

            error += (y[i] - prediction).powi(2);
            count += 1;
        }

        Ok(if count > 0 {
            error / count as Float
        } else {
            1.0
        })
    }

    fn compute_information_criteria(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        max_lag: usize,
    ) -> SklResult<HashMap<String, Array1<Float>>> {
        let mut aic_values = Array1::zeros(max_lag + 1);
        let mut bic_values = Array1::zeros(max_lag + 1);

        for lag in 1..=max_lag {
            let prediction_error = self.compute_prediction_error(y, Some(x), lag)?;
            let n = (y.len() - lag) as Float;
            let k = (2 * lag) as Float; // Number of parameters

            // AIC = 2k + n*ln(RSS/n)
            aic_values[lag] = 2.0 * k + n * (prediction_error).ln();

            // BIC = k*ln(n) + n*ln(RSS/n)
            bic_values[lag] = k * n.ln() + n * (prediction_error).ln();
        }

        let mut criteria = HashMap::new();
        criteria.insert("AIC".to_string(), aic_values);
        criteria.insert("BIC".to_string(), bic_values);

        Ok(criteria)
    }

    fn compute_dtw(
        &self,
        query: &ArrayView1<Float>,
        reference: &ArrayView1<Float>,
    ) -> SklResult<(Float, Array2<usize>, Array2<Float>)> {
        let n = query.len();
        let m = reference.len();

        // Initialize distance matrix
        let mut distance_matrix = Array2::from_elem((n, m), Float::INFINITY);

        // Compute local distances
        for i in 0..n {
            for j in 0..m {
                distance_matrix[[i, j]] = (query[i] - reference[j]).abs();
            }
        }

        // Compute cumulative distance matrix
        let mut cumulative = Array2::from_elem((n, m), Float::INFINITY);
        cumulative[[0, 0]] = distance_matrix[[0, 0]];

        // Initialize first row and column
        for i in 1..n {
            cumulative[[i, 0]] = distance_matrix[[i, 0]] + cumulative[[i - 1, 0]];
        }
        for j in 1..m {
            cumulative[[0, j]] = distance_matrix[[0, j]] + cumulative[[0, j - 1]];
        }

        // Fill cumulative matrix
        for i in 1..n {
            for j in 1..m {
                let cost = distance_matrix[[i, j]];
                let min_prev = cumulative[[i - 1, j]]
                    .min(cumulative[[i, j - 1]])
                    .min(cumulative[[i - 1, j - 1]]);
                cumulative[[i, j]] = cost + min_prev;
            }
        }

        // Backtrack to find optimal path
        let mut warping_path = Vec::new();
        let mut i = n - 1;
        let mut j = m - 1;

        warping_path.push((i, j));

        while i > 0 || j > 0 {
            if i == 0 {
                j -= 1;
            } else if j == 0 {
                i -= 1;
            } else {
                let diag = cumulative[[i - 1, j - 1]];
                let left = cumulative[[i, j - 1]];
                let up = cumulative[[i - 1, j]];

                if diag <= left && diag <= up {
                    i -= 1;
                    j -= 1;
                } else if left <= up {
                    j -= 1;
                } else {
                    i -= 1;
                }
            }
            warping_path.push((i, j));
        }

        warping_path.reverse();

        // Convert path to Array2
        let path_len = warping_path.len();
        let mut path_array = Array2::zeros((path_len, 2));
        for (idx, &(i, j)) in warping_path.iter().enumerate() {
            path_array[[idx, 0]] = i;
            path_array[[idx, 1]] = j;
        }

        let final_distance = cumulative[[n - 1, m - 1]];

        Ok((final_distance, path_array, distance_matrix))
    }

    fn create_alignment_explanation(
        &self,
        warping_path: &Array2<usize>,
        distance_matrix: &Array2<Float>,
    ) -> SklResult<Vec<AlignmentSegment>> {
        let mut segments = Vec::new();
        let path_len = warping_path.nrows();

        if path_len == 0 {
            return Ok(segments);
        }

        let mut current_segment_start = 0;
        let mut current_query_start = warping_path[[0, 0]];
        let mut current_ref_start = warping_path[[0, 1]];

        for i in 1..path_len {
            let prev_query = warping_path[[i - 1, 0]];
            let prev_ref = warping_path[[i - 1, 1]];
            let curr_query = warping_path[[i, 0]];
            let curr_ref = warping_path[[i, 1]];

            // Detect alignment type change
            let prev_type = self.get_alignment_type(prev_query, prev_ref, curr_query, curr_ref);

            if i == path_len - 1 || i % 10 == 0 {
                // Create segments periodically
                let end_query = curr_query;
                let end_ref = curr_ref;

                // Compute average cost for this segment
                let mut total_cost = 0.0;
                let mut cost_count = 0;

                for j in current_segment_start..=i {
                    if j < path_len {
                        let q_idx = warping_path[[j, 0]];
                        let r_idx = warping_path[[j, 1]];
                        if q_idx < distance_matrix.nrows() && r_idx < distance_matrix.ncols() {
                            total_cost += distance_matrix[[q_idx, r_idx]];
                            cost_count += 1;
                        }
                    }
                }

                let avg_cost = if cost_count > 0 {
                    total_cost / cost_count as Float
                } else {
                    0.0
                };

                segments.push(AlignmentSegment {
                    query_range: (current_query_start, end_query),
                    reference_range: (current_ref_start, end_ref),
                    alignment_cost: avg_cost,
                    alignment_type: prev_type,
                });

                current_segment_start = i;
                current_query_start = curr_query;
                current_ref_start = curr_ref;
            }
        }

        Ok(segments)
    }

    fn get_alignment_type(
        &self,
        prev_q: usize,
        prev_r: usize,
        curr_q: usize,
        curr_r: usize,
    ) -> AlignmentType {
        let q_diff = curr_q.saturating_sub(prev_q);
        let r_diff = curr_r.saturating_sub(prev_r);

        if q_diff == 1 && r_diff == 1 {
            AlignmentType::Match
        } else if q_diff == 1 && r_diff > 1 {
            AlignmentType::Expansion
        } else if q_diff > 1 && r_diff == 1 {
            AlignmentType::Compression
        } else {
            AlignmentType::Match // Default
        }
    }

    fn compute_dtw_feature_importance(
        &self,
        query: &ArrayView1<Float>,
        reference: &ArrayView1<Float>,
        warping_path: &Array2<usize>,
    ) -> SklResult<Array1<Float>> {
        let n = query.len();
        let mut importance = Array1::zeros(n);
        let path_len = warping_path.nrows();

        // Count how often each query point is used in the alignment
        let mut usage_count = Array1::<Float>::zeros(n);
        let mut total_cost = Array1::<Float>::zeros(n);

        for i in 0..path_len {
            let q_idx = warping_path[[i, 0]];
            let r_idx = warping_path[[i, 1]];

            if q_idx < n && r_idx < reference.len() {
                usage_count[q_idx] += 1.0;
                total_cost[q_idx] += (query[q_idx] - reference[r_idx]).abs();
            }
        }

        // Importance based on usage frequency and alignment cost
        for i in 0..n {
            if usage_count[i] > 0.0 {
                importance[i] = usage_count[i] * (1.0 / (total_cost[i] / usage_count[i] + 1e-10));
            }
        }

        // Normalize
        let max_importance = importance.iter().cloned().fold(0.0, Float::max);
        if max_importance > 0.0 {
            importance /= max_importance;
        }

        Ok(importance)
    }

    fn normalize_warping_path(
        &self,
        warping_path: &Array2<usize>,
        query_len: usize,
        reference_len: usize,
    ) -> SklResult<Array2<Float>> {
        let path_len = warping_path.nrows();
        let mut normalized_path = Array2::zeros((path_len, 2));

        for i in 0..path_len {
            normalized_path[[i, 0]] = warping_path[[i, 0]] as Float / query_len as Float;
            normalized_path[[i, 1]] = warping_path[[i, 1]] as Float / reference_len as Float;
        }

        Ok(normalized_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_time_series_analyzer_creation() {
        let analyzer = TimeSeriesAnalyzer::default();
        assert_eq!(analyzer.config.max_lag, 20);
        assert_eq!(analyzer.config.trend_window, 5);
    }

    #[test]
    fn test_seasonal_decomposition() {
        let analyzer = TimeSeriesAnalyzer::default();

        // Create simple seasonal time series
        let mut series = Array1::zeros(48);
        for i in 0..48 {
            series[i] = (i as Float * 0.1)
                + (2.0 * std::f64::consts::PI * i as Float / 12.0).sin() as Float;
        }

        let result = analyzer.seasonal_decomposition(&series.view());
        assert!(result.is_ok());

        let decomp = result.unwrap();
        assert_eq!(decomp.original.len(), 48);
        assert_eq!(decomp.trend.len(), 48);
        assert_eq!(decomp.seasonal.len(), 48);
        assert_eq!(decomp.residual.len(), 48);
        assert!(decomp.trend_strength >= 0.0 && decomp.trend_strength <= 1.0);
        assert!(decomp.seasonal_strength >= 0.0 && decomp.seasonal_strength <= 1.0);
    }

    #[test]
    fn test_trend_analysis() {
        let analyzer = TimeSeriesAnalyzer::default();

        // Create trending time series
        let series: Array1<Float> = (0..20)
            .map(|i| i as Float + 0.1 * (i as Float).powi(2))
            .collect();

        let result = analyzer.trend_analysis(&series.view());
        assert!(result.is_ok());

        let trend = result.unwrap();
        assert!(matches!(trend.trend_direction, TrendDirection::Increasing));
        assert!(trend.trend_magnitude > 0.0);
        assert!(!trend.trend_segments.is_empty());
    }

    #[test]
    fn test_lag_importance() {
        let analyzer = TimeSeriesAnalyzer::default();

        // Create time series with clear lag relationship
        let n = 50;
        let series1: Array1<Float> = (0..n).map(|i| (i as Float * 0.1).sin()).collect();
        let series2: Array1<Float> = (0..n)
            .map(|i| if i >= 2 { series1[i - 2] } else { 0.0 })
            .collect();

        let result = analyzer.lag_importance(&series1.view(), &series2.view());
        assert!(result.is_ok());

        let lag_imp = result.unwrap();
        assert_eq!(
            lag_imp.lag_scores.len(),
            analyzer.config.max_lag.min(n / 4) + 1
        );
        assert_eq!(
            lag_imp.cross_correlations.len(),
            analyzer.config.max_lag.min(n / 4) + 1
        );
        assert!(lag_imp.optimal_lag <= analyzer.config.max_lag);
    }

    #[test]
    fn test_dtw_explanation() {
        let analyzer = TimeSeriesAnalyzer::default();

        // Create similar time series with slight offset
        let query: Array1<Float> = (0..10).map(|i| (i as Float * 0.5).sin()).collect();
        let reference: Array1<Float> = (0..12).map(|i| ((i + 1) as Float * 0.5).sin()).collect();

        let result = analyzer.dtw_explanation(&query.view(), &reference.view());
        assert!(result.is_ok());

        let dtw = result.unwrap();
        assert!(dtw.distance >= 0.0);
        assert!(dtw.warping_path.nrows() > 0);
        assert_eq!(dtw.warping_path.ncols(), 2);
        assert!(!dtw.alignment_explanation.is_empty());
        assert_eq!(dtw.dtw_feature_importance.len(), query.len());
    }

    #[test]
    fn test_moving_average() {
        let analyzer = TimeSeriesAnalyzer::default();
        let series = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result = analyzer.compute_moving_average(&series.view(), 3);
        assert!(result.is_ok());

        let ma = result.unwrap();
        assert_eq!(ma.len(), series.len());
        assert!(ma.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_autocorrelation() {
        let analyzer = TimeSeriesAnalyzer::default();
        let series = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // Alternating pattern

        let result = analyzer.compute_autocorrelation(&series.view(), 2);
        assert!(result.is_ok());

        let ac = result.unwrap();
        // Should have high autocorrelation at lag 2 due to pattern
        assert!(ac > 0.5);
    }

    #[test]
    fn test_cross_correlation() {
        let analyzer = TimeSeriesAnalyzer::default();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 3.0, 4.0, 5.0, 6.0]; // y = x + 1 with shift

        let result = analyzer.compute_cross_correlation(&x.view(), &y.view(), 2);
        assert!(result.is_ok());

        let cc = result.unwrap();
        assert_eq!(cc.len(), 3); // lag 0, 1, 2
        assert!(cc[0] > 0.8); // High correlation at lag 0
    }

    #[test]
    fn test_change_point_detection() {
        let analyzer = TimeSeriesAnalyzer::default();

        // Create series with obvious change point
        let mut series = Array1::zeros(20);
        for i in 0..10 {
            series[i] = 1.0;
        }
        for i in 10..20 {
            series[i] = 5.0;
        }

        let result = analyzer.detect_change_points(&series.view());
        assert!(result.is_ok());

        let change_points = result.unwrap();
        // Should detect change point around index 10
        assert!(!change_points.is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_temporal_importance_small_data() {
        let analyzer = TimeSeriesAnalyzer::default();

        // Create small dataset
        let X = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as Float);
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Simple linear model function
        let model_fn =
            |X: &ArrayView2<Float>| -> SklResult<Array1<Float>> { Ok(X.column(0).to_owned()) };

        let result = analyzer.temporal_importance(&X.view(), &y.view(), &model_fn);
        assert!(result.is_ok());

        let temp_imp = result.unwrap();
        assert_eq!(temp_imp.temporal_scores.len(), 5);
        assert_eq!(temp_imp.significance_map.len(), 5);
        assert_eq!(temp_imp.confidence_intervals.nrows(), 5);
        assert_eq!(temp_imp.confidence_intervals.ncols(), 2);
    }
}
