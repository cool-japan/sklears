//! Temporal Data Structures for Time Series Tree Algorithms
//!
//! This module provides specialized data structures and algorithms for handling
//! time-ordered data in tree-based models, including temporal splits, time-aware
//! tree construction, and forecasting-specific optimizations.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::Untrained,
};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::decision_tree::{DecisionTreeConfig, SplitCriterion};

/// Time-aware split criteria for temporal data
#[derive(Debug, Clone)]
pub enum TemporalSplitCriterion {
    /// Standard split criteria applied to temporal features
    Standard(SplitCriterion),
    /// Time-aware MSE that considers temporal autocorrelation
    TemporalMSE { autocorr_penalty: f64, lag: usize },
    /// Trend-aware splitting that preserves trend information
    TrendAware {
        trend_weight: f64,
        min_trend_length: usize,
    },
    /// Seasonal decomposition-aware splitting
    SeasonalAware {
        seasonal_period: usize,
        seasonal_weight: f64,
    },
    /// Change point detection splitting
    ChangePoint {
        change_penalty: f64,
        min_segment_length: usize,
    },
    /// Regime-aware splitting for regime changes
    RegimeAware {
        regime_window: usize,
        regime_threshold: f64,
    },
}

/// Temporal features extracted from time series data
#[derive(Debug, Clone)]
pub struct TemporalFeatures {
    /// Lag features (previous values)
    pub lag_features: Vec<usize>,
    /// Rolling window statistics
    pub rolling_windows: Vec<RollingWindow>,
    /// Trend features
    pub trend_features: Vec<TrendFeature>,
    /// Seasonal features
    pub seasonal_features: Vec<SeasonalFeature>,
    /// Change point indicators
    pub change_points: Vec<usize>,
    /// Regime indicators
    pub regimes: Vec<usize>,
}

/// Rolling window statistics configuration
#[derive(Debug, Clone)]
pub struct RollingWindow {
    /// Window size
    pub window_size: usize,
    /// Statistics to compute
    pub statistics: Vec<WindowStatistic>,
    /// Minimum periods required
    pub min_periods: usize,
}

/// Window statistics to compute
#[derive(Debug, Clone, Copy)]
pub enum WindowStatistic {
    Mean,
    Std,
    Min,
    Max,
    Median,
    Quantile(f64),
    Skewness,
    Kurtosis,
    AutoCorrelation(usize),
}

/// Trend feature configuration
#[derive(Debug, Clone)]
pub struct TrendFeature {
    /// Trend detection method
    pub method: TrendMethod,
    /// Window size for trend calculation
    pub window_size: usize,
    /// Minimum trend strength
    pub min_strength: f64,
}

/// Trend detection methods
#[derive(Debug, Clone, Copy)]
pub enum TrendMethod {
    /// Linear regression slope
    LinearSlope,
    /// Mann-Kendall trend test
    MannKendall,
    /// Moving average trend
    MovingAverage,
    /// Hodrick-Prescott filter
    HodrickPrescott { lambda: f64 },
}

/// Seasonal feature configuration
#[derive(Debug, Clone)]
pub struct SeasonalFeature {
    /// Seasonal period (e.g., 24 for hourly data with daily seasonality)
    pub period: usize,
    /// Decomposition method
    pub method: SeasonalMethod,
    /// Number of harmonics to include
    pub n_harmonics: usize,
}

/// Seasonal decomposition methods
#[derive(Debug, Clone, Copy)]
pub enum SeasonalMethod {
    /// Classical additive decomposition
    Additive,
    /// Classical multiplicative decomposition
    Multiplicative,
    /// STL (Seasonal and Trend decomposition using Loess)
    STL { seasonal_window: usize },
    /// Fourier series decomposition
    Fourier,
    /// X-11 seasonal adjustment
    X11,
}

/// Temporal data structure for time series
#[derive(Debug, Clone)]
pub struct TemporalDataStructure {
    /// Time index (timestamps or sequential indices)
    pub time_index: Array1<f64>,
    /// Time series values
    pub values: Array2<f64>,
    /// Temporal features
    pub temporal_features: TemporalFeatures,
    /// Temporal metadata
    pub metadata: TemporalMetadata,
}

/// Metadata for temporal data
#[derive(Debug, Clone)]
pub struct TemporalMetadata {
    /// Sampling frequency (e.g., daily, hourly)
    pub frequency: Frequency,
    /// Start time
    pub start_time: Option<f64>,
    /// End time
    pub end_time: Option<f64>,
    /// Missing value indicators
    pub missing_values: Vec<bool>,
    /// Outlier indicators
    pub outliers: Vec<bool>,
    /// Structural breaks
    pub structural_breaks: Vec<usize>,
}

/// Time series frequency
#[derive(Debug, Clone, Copy)]
pub enum Frequency {
    /// Milliseconds
    Millisecond,
    /// Seconds
    Second,
    /// Minutes
    Minute,
    /// Hours
    Hour,
    /// Days
    Day,
    /// Weeks
    Week,
    /// Months
    Month,
    /// Quarters
    Quarter,
    /// Years
    Year,
    /// Custom frequency (in seconds)
    Custom(f64),
}

impl TemporalDataStructure {
    pub fn new(time_index: Array1<f64>, values: Array2<f64>, frequency: Frequency) -> Self {
        let n_samples = time_index.len();

        Self {
            time_index,
            values,
            temporal_features: TemporalFeatures {
                lag_features: Vec::new(),
                rolling_windows: Vec::new(),
                trend_features: Vec::new(),
                seasonal_features: Vec::new(),
                change_points: Vec::new(),
                regimes: Vec::new(),
            },
            metadata: TemporalMetadata {
                frequency,
                start_time: None,
                end_time: None,
                missing_values: vec![false; n_samples],
                outliers: vec![false; n_samples],
                structural_breaks: Vec::new(),
            },
        }
    }

    /// Extract lag features
    pub fn extract_lag_features(&mut self, lags: Vec<usize>) -> Result<Array2<f64>> {
        let n_samples = self.values.nrows();
        let n_features = self.values.ncols();
        let max_lag = *lags.iter().max().unwrap_or(&0);

        if max_lag >= n_samples {
            return Err(SklearsError::InvalidInput(
                "Lag exceeds data length".to_string(),
            ));
        }

        let n_lag_features = lags.len() * n_features;
        let mut lag_features = Array2::zeros((n_samples, n_lag_features));

        let mut feature_idx = 0;
        for &lag in &lags {
            for col in 0..n_features {
                for row in lag..n_samples {
                    lag_features[[row, feature_idx]] = self.values[[row - lag, col]];
                }
                feature_idx += 1;
            }
        }

        self.temporal_features.lag_features = lags;
        Ok(lag_features)
    }

    /// Extract rolling window features
    pub fn extract_rolling_features(&mut self, windows: Vec<RollingWindow>) -> Result<Array2<f64>> {
        let n_samples = self.values.nrows();
        let n_features = self.values.ncols();

        let mut total_features = 0;
        for window in &windows {
            total_features += window.statistics.len() * n_features;
        }

        let mut rolling_features = Array2::zeros((n_samples, total_features));
        let mut feature_idx = 0;

        for window in &windows {
            for col in 0..n_features {
                for stat in &window.statistics {
                    for row in window.window_size..n_samples {
                        let window_start = if row >= window.window_size {
                            row - window.window_size
                        } else {
                            0
                        };

                        let window_data: Vec<f64> = (window_start..=row)
                            .map(|i| self.values[[i, col]])
                            .collect();

                        let stat_value = self.calculate_window_statistic(&window_data, *stat)?;
                        rolling_features[[row, feature_idx]] = stat_value;
                    }
                    feature_idx += 1;
                }
            }
        }

        self.temporal_features.rolling_windows = windows;
        Ok(rolling_features)
    }

    /// Calculate window statistic
    fn calculate_window_statistic(&self, data: &[f64], stat: WindowStatistic) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        match stat {
            WindowStatistic::Mean => Ok(data.iter().sum::<f64>() / data.len() as f64),
            WindowStatistic::Std => {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let variance =
                    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
                Ok(variance.sqrt())
            }
            WindowStatistic::Min => Ok(data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x))),
            WindowStatistic::Max => Ok(data.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))),
            WindowStatistic::Median => {
                let mut sorted_data = data.to_vec();
                sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted_data.len();
                if n % 2 == 0 {
                    Ok((sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0)
                } else {
                    Ok(sorted_data[n / 2])
                }
            }
            WindowStatistic::Quantile(q) => {
                let mut sorted_data = data.to_vec();
                sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let index = (q * (sorted_data.len() - 1) as f64) as usize;
                Ok(sorted_data.get(index).copied().unwrap_or(0.0))
            }
            WindowStatistic::Skewness => {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let std = {
                    let variance =
                        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
                    variance.sqrt()
                };

                if std == 0.0 {
                    Ok(0.0)
                } else {
                    let skewness = data
                        .iter()
                        .map(|&x| ((x - mean) / std).powi(3))
                        .sum::<f64>()
                        / data.len() as f64;
                    Ok(skewness)
                }
            }
            WindowStatistic::Kurtosis => {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let std = {
                    let variance =
                        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
                    variance.sqrt()
                };

                if std == 0.0 {
                    Ok(0.0)
                } else {
                    let kurtosis = data
                        .iter()
                        .map(|&x| ((x - mean) / std).powi(4))
                        .sum::<f64>()
                        / data.len() as f64;
                    Ok(kurtosis - 3.0) // Excess kurtosis
                }
            }
            WindowStatistic::AutoCorrelation(lag) => {
                if data.len() <= lag {
                    Ok(0.0)
                } else {
                    let n = data.len() - lag;
                    let x1: Vec<f64> = data[..n].to_vec();
                    let x2: Vec<f64> = data[lag..].to_vec();

                    let mean1 = x1.iter().sum::<f64>() / x1.len() as f64;
                    let mean2 = x2.iter().sum::<f64>() / x2.len() as f64;

                    let numerator: f64 = x1
                        .iter()
                        .zip(x2.iter())
                        .map(|(&a, &b)| (a - mean1) * (b - mean2))
                        .sum();

                    let denom1: f64 = x1.iter().map(|&x| (x - mean1).powi(2)).sum();
                    let denom2: f64 = x2.iter().map(|&x| (x - mean2).powi(2)).sum();

                    let denominator = (denom1 * denom2).sqrt();

                    if denominator == 0.0 {
                        Ok(0.0)
                    } else {
                        Ok(numerator / denominator)
                    }
                }
            }
        }
    }

    /// Detect trend using specified method
    pub fn detect_trend(&mut self, features: Vec<TrendFeature>) -> Result<Array2<f64>> {
        let n_samples = self.values.nrows();
        let n_features = self.values.ncols();

        let mut trend_features = Array2::zeros((n_samples, features.len() * n_features));
        let mut feature_idx = 0;

        for trend_feature in &features {
            for col in 0..n_features {
                for row in trend_feature.window_size..n_samples {
                    let window_start = row.saturating_sub(trend_feature.window_size);
                    let window_data: Vec<f64> = (window_start..=row)
                        .map(|i| self.values[[i, col]])
                        .collect();

                    let trend_value = self.calculate_trend(&window_data, trend_feature.method)?;
                    trend_features[[row, feature_idx]] = trend_value;
                }
                feature_idx += 1;
            }
        }

        self.temporal_features.trend_features = features;
        Ok(trend_features)
    }

    /// Calculate trend statistic
    fn calculate_trend(&self, data: &[f64], method: TrendMethod) -> Result<f64> {
        match method {
            TrendMethod::LinearSlope => {
                if data.len() < 2 {
                    return Ok(0.0);
                }

                let n = data.len() as f64;
                let x_mean = (n - 1.0) / 2.0;
                let y_mean = data.iter().sum::<f64>() / n;

                let numerator: f64 = data
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
                    .sum();

                let denominator: f64 = (0..data.len()).map(|i| (i as f64 - x_mean).powi(2)).sum();

                if denominator == 0.0 {
                    Ok(0.0)
                } else {
                    Ok(numerator / denominator)
                }
            }
            TrendMethod::MannKendall => {
                // Simplified Mann-Kendall test statistic
                let mut s = 0i32;
                for i in 0..data.len() {
                    for j in (i + 1)..data.len() {
                        if data[j] > data[i] {
                            s += 1;
                        } else if data[j] < data[i] {
                            s -= 1;
                        }
                    }
                }
                Ok(s as f64)
            }
            TrendMethod::MovingAverage => {
                if data.len() < 2 {
                    return Ok(0.0);
                }

                let first_half_mean =
                    data[..data.len() / 2].iter().sum::<f64>() / (data.len() / 2) as f64;
                let second_half_mean = data[data.len() / 2..].iter().sum::<f64>()
                    / (data.len() - data.len() / 2) as f64;

                Ok(second_half_mean - first_half_mean)
            }
            TrendMethod::HodrickPrescott { lambda } => {
                // Simplified HP filter (just return linear trend for now)
                self.calculate_trend(data, TrendMethod::LinearSlope)
            }
        }
    }

    /// Detect seasonal patterns
    pub fn detect_seasonality(&mut self, features: Vec<SeasonalFeature>) -> Result<Array2<f64>> {
        let n_samples = self.values.nrows();
        let n_features = self.values.ncols();

        let mut seasonal_features = Array2::zeros((n_samples, features.len() * n_features));
        let mut feature_idx = 0;

        for seasonal_feature in &features {
            for col in 0..n_features {
                let seasonal_values = self
                    .calculate_seasonality(&self.values.column(col).to_vec(), seasonal_feature)?;

                for (row, &value) in seasonal_values.iter().enumerate() {
                    if row < n_samples {
                        seasonal_features[[row, feature_idx]] = value;
                    }
                }
                feature_idx += 1;
            }
        }

        self.temporal_features.seasonal_features = features;
        Ok(seasonal_features)
    }

    /// Calculate seasonal component
    fn calculate_seasonality(&self, data: &[f64], feature: &SeasonalFeature) -> Result<Vec<f64>> {
        match feature.method {
            SeasonalMethod::Additive => self.additive_decomposition(data, feature.period),
            SeasonalMethod::Multiplicative => {
                self.multiplicative_decomposition(data, feature.period)
            }
            SeasonalMethod::Fourier => {
                self.fourier_decomposition(data, feature.period, feature.n_harmonics)
            }
            _ => {
                // Fallback to simple seasonal average
                self.simple_seasonal_decomposition(data, feature.period)
            }
        }
    }

    /// Simple additive seasonal decomposition
    fn additive_decomposition(&self, data: &[f64], period: usize) -> Result<Vec<f64>> {
        let mut seasonal_component = vec![0.0; data.len()];

        if period == 0 || data.len() < period {
            return Ok(seasonal_component);
        }

        // Calculate seasonal averages
        let mut seasonal_sums = vec![0.0; period];
        let mut seasonal_counts = vec![0; period];

        for (i, &value) in data.iter().enumerate() {
            let season_idx = i % period;
            seasonal_sums[season_idx] += value;
            seasonal_counts[season_idx] += 1;
        }

        let seasonal_averages: Vec<f64> = seasonal_sums
            .iter()
            .zip(seasonal_counts.iter())
            .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
            .collect();

        // Apply seasonal component
        for (i, seasonal_val) in seasonal_component.iter_mut().enumerate() {
            let season_idx = i % period;
            *seasonal_val = seasonal_averages[season_idx];
        }

        Ok(seasonal_component)
    }

    /// Simple multiplicative seasonal decomposition
    fn multiplicative_decomposition(&self, data: &[f64], period: usize) -> Result<Vec<f64>> {
        // Similar to additive but with ratios
        let additive_seasonal = self.additive_decomposition(data, period)?;
        let overall_mean = data.iter().sum::<f64>() / data.len() as f64;

        let multiplicative_seasonal: Vec<f64> = additive_seasonal
            .iter()
            .map(|&seasonal| seasonal / overall_mean.max(1e-8))
            .collect();

        Ok(multiplicative_seasonal)
    }

    /// Fourier-based seasonal decomposition
    fn fourier_decomposition(
        &self,
        data: &[f64],
        period: usize,
        n_harmonics: usize,
    ) -> Result<Vec<f64>> {
        let mut seasonal_component = vec![0.0; data.len()];

        if period == 0 || data.len() < period {
            return Ok(seasonal_component);
        }

        let fundamental_freq = 2.0 * std::f64::consts::PI / period as f64;

        for harmonic in 1..=n_harmonics {
            let freq = harmonic as f64 * fundamental_freq;

            for (i, seasonal_val) in seasonal_component.iter_mut().enumerate() {
                *seasonal_val += (freq * i as f64).sin() + (freq * i as f64).cos();
            }
        }

        Ok(seasonal_component)
    }

    /// Simple seasonal decomposition
    fn simple_seasonal_decomposition(&self, data: &[f64], period: usize) -> Result<Vec<f64>> {
        self.additive_decomposition(data, period)
    }

    /// Detect change points
    pub fn detect_change_points(
        &mut self,
        min_segment_length: usize,
        penalty: f64,
    ) -> Result<Vec<usize>> {
        let mut change_points = Vec::new();

        for col in 0..self.values.ncols() {
            let data = self.values.column(col).to_vec();
            let col_change_points =
                self.detect_change_points_in_series(&data, min_segment_length, penalty)?;
            change_points.extend(col_change_points);
        }

        change_points.sort_unstable();
        change_points.dedup();

        self.temporal_features.change_points = change_points.clone();
        Ok(change_points)
    }

    /// Detect change points in a single time series
    fn detect_change_points_in_series(
        &self,
        data: &[f64],
        min_segment_length: usize,
        penalty: f64,
    ) -> Result<Vec<usize>> {
        let mut change_points = Vec::new();

        if data.len() < 2 * min_segment_length {
            return Ok(change_points);
        }

        // Simple change point detection using sliding window means
        for i in min_segment_length..(data.len() - min_segment_length) {
            let left_window = &data[(i.saturating_sub(min_segment_length))..i];
            let right_window = &data[i..(i + min_segment_length).min(data.len())];

            let left_mean = left_window.iter().sum::<f64>() / left_window.len() as f64;
            let right_mean = right_window.iter().sum::<f64>() / right_window.len() as f64;

            let change_magnitude = (left_mean - right_mean).abs();

            if change_magnitude > penalty {
                change_points.push(i);
            }
        }

        Ok(change_points)
    }
}

/// Temporal Decision Tree for time series data
pub struct TemporalDecisionTree<State = Untrained> {
    /// Base tree configuration
    config: DecisionTreeConfig,
    /// Temporal-specific configuration
    temporal_config: TemporalTreeConfig,
    /// Temporal data structure
    temporal_data: Option<TemporalDataStructure>,
    /// Trained model
    model: Option<Box<dyn TemporalTreeModel>>,
    /// State marker
    state: PhantomData<State>,
}

/// Configuration for temporal decision trees
#[derive(Debug, Clone)]
pub struct TemporalTreeConfig {
    /// Temporal split criterion
    pub split_criterion: TemporalSplitCriterion,
    /// Maximum lookback window
    pub max_lookback: usize,
    /// Minimum forecast horizon
    pub min_forecast_horizon: usize,
    /// Enable temporal regularization
    pub temporal_regularization: bool,
    /// Regularization strength
    pub regularization_strength: f64,
    /// Enable cross-temporal validation
    pub cross_temporal_validation: bool,
}

impl Default for TemporalTreeConfig {
    fn default() -> Self {
        Self {
            split_criterion: TemporalSplitCriterion::Standard(SplitCriterion::MSE),
            max_lookback: 10,
            min_forecast_horizon: 1,
            temporal_regularization: false,
            regularization_strength: 0.01,
            cross_temporal_validation: false,
        }
    }
}

/// Trait for temporal tree models
pub trait TemporalTreeModel: Send + Sync {
    /// Predict future values
    fn predict(&self, x: &Array2<f64>, forecast_horizon: usize) -> Result<Array1<f64>>;

    /// Update model with new temporal data
    fn update(&mut self, temporal_data: &TemporalDataStructure) -> Result<()>;

    /// Get temporal feature importance
    fn get_temporal_importance(&self) -> Result<HashMap<String, f64>>;
}

/// Simple temporal tree implementation
#[derive(Debug, Clone)]
pub struct SimpleTemporalTree {
    /// Tree nodes
    nodes: Vec<TemporalTreeNode>,
    /// Configuration
    config: TemporalTreeConfig,
    /// Temporal feature importance
    temporal_importance: HashMap<String, f64>,
}

/// Temporal tree node
#[derive(Debug, Clone)]
pub struct TemporalTreeNode {
    /// Node ID
    pub id: usize,
    /// Feature index (can be temporal feature)
    pub feature_idx: Option<usize>,
    /// Split threshold
    pub threshold: Option<f64>,
    /// Temporal split type
    pub temporal_split_type: Option<TemporalSplitType>,
    /// Prediction value
    pub prediction: f64,
    /// Temporal prediction components
    pub temporal_components: TemporalComponents,
    /// Child nodes
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    /// Is leaf
    pub is_leaf: bool,
    /// Node depth
    pub depth: usize,
}

/// Types of temporal splits
#[derive(Debug, Clone)]
pub enum TemporalSplitType {
    /// Standard feature split
    Standard,
    /// Time-based split
    Temporal { time_threshold: f64 },
    /// Lag-based split
    Lag { lag: usize, threshold: f64 },
    /// Trend-based split
    Trend { trend_type: TrendMethod },
    /// Seasonal split
    Seasonal { period: usize, phase: f64 },
}

/// Temporal prediction components
#[derive(Debug, Clone)]
pub struct TemporalComponents {
    /// Trend component
    pub trend: f64,
    /// Seasonal component
    pub seasonal: f64,
    /// Residual component
    pub residual: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

impl Default for TemporalComponents {
    fn default() -> Self {
        Self {
            trend: 0.0,
            seasonal: 0.0,
            residual: 0.0,
            confidence_interval: (0.0, 0.0),
        }
    }
}
