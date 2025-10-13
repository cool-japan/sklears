//! Time Series Ensemble Methods
//!
//! This module provides specialized ensemble methods for time series forecasting
//! and analysis, including temporal ensemble construction, drift adaptation,
//! and time-aware model selection.

use crate::gradient_boosting::{GradientBoostingRegressor, TrainedGradientBoostingRegressor};
use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{
    error::Result as SklResult,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict},
};
use std::f64;

/// Configuration for time series ensemble methods
#[derive(Debug, Clone)]
pub struct TimeSeriesEnsembleConfig {
    /// Number of time steps to look back for feature engineering
    pub window_size: usize,
    /// Number of models in the ensemble
    pub n_estimators: usize,
    /// Temporal overlap between ensemble members
    pub temporal_overlap: f64,
    /// Whether to use seasonal decomposition
    pub use_seasonal_decomposition: bool,
    /// Number of seasonal components to extract
    pub seasonal_periods: Vec<usize>,
    /// Temporal weight decay for older observations
    pub temporal_decay: f64,
    /// Strategy for handling concept drift
    pub drift_adaptation: DriftAdaptationStrategy,
    /// Cross-validation strategy for time series
    pub cv_strategy: TimeSeriesCVStrategy,
    /// Ensemble aggregation method
    pub aggregation_method: TemporalAggregationMethod,
}

impl Default for TimeSeriesEnsembleConfig {
    fn default() -> Self {
        Self {
            window_size: 24,
            n_estimators: 10,
            temporal_overlap: 0.5,
            use_seasonal_decomposition: true,
            seasonal_periods: vec![7, 30, 365], // Daily, monthly, yearly patterns
            temporal_decay: 0.95,
            drift_adaptation: DriftAdaptationStrategy::SlidingWindow,
            cv_strategy: TimeSeriesCVStrategy::TimeSeriesSplit,
            aggregation_method: TemporalAggregationMethod::WeightedAverage,
        }
    }
}

/// Strategies for adapting to concept drift in time series
#[derive(Debug, Clone, PartialEq)]
pub enum DriftAdaptationStrategy {
    /// Use a sliding window approach
    SlidingWindow,
    /// Exponential forgetting of old observations
    ExponentialForgetting,
    /// Dynamic ensemble size based on performance
    DynamicEnsemble,
    /// Online update of ensemble weights
    OnlineWeightUpdate,
    /// Seasonal pattern adaptation
    SeasonalAdaptation,
}

/// Cross-validation strategies for time series
#[derive(Debug, Clone, PartialEq)]
pub enum TimeSeriesCVStrategy {
    /// Traditional time series split (no shuffling)
    TimeSeriesSplit,
    /// Blocked cross-validation for time series
    BlockedCV,
    /// Purged time series split (gap between train/test)
    PurgedTimeSeriesSplit,
    /// Walk-forward validation
    WalkForward,
    /// Time series sliding window validation
    SlidingWindow,
}

/// Methods for aggregating temporal ensemble predictions
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalAggregationMethod {
    /// Simple average across time
    SimpleAverage,
    /// Weighted average with temporal decay
    WeightedAverage,
    /// Median aggregation for robustness
    MedianAggregation,
    /// Exponential smoothing of predictions
    ExponentialSmoothing,
    /// Kalman filter-based aggregation
    KalmanFilter,
    /// Bayesian model averaging with temporal priors
    BayesianTemporal,
}

/// Time series ensemble classifier
pub struct TimeSeriesEnsembleClassifier {
    config: TimeSeriesEnsembleConfig,
    base_models: Vec<TrainedGradientBoostingRegressor>,
    temporal_weights: Vec<f64>,
    seasonal_components: Option<SeasonalComponents>,
    drift_detector: Option<Box<dyn ConceptDriftDetector>>,
    is_fitted: bool,
}

/// Time series ensemble regressor
pub struct TimeSeriesEnsembleRegressor {
    config: TimeSeriesEnsembleConfig,
    base_models: Vec<TrainedGradientBoostingRegressor>,
    temporal_weights: Vec<f64>,
    seasonal_components: Option<SeasonalComponents>,
    drift_detector: Option<Box<dyn ConceptDriftDetector>>,
    is_fitted: bool,
}

/// Seasonal decomposition components
#[derive(Debug, Clone)]
pub struct SeasonalComponents {
    /// Trend component
    pub trend: Vec<f64>,
    /// Seasonal component for each period
    pub seasonal: Vec<Vec<f64>>,
    /// Residual component
    pub residual: Vec<f64>,
}

/// Trait for concept drift detection in time series
pub trait ConceptDriftDetector: Send + Sync {
    /// Detect drift given new observations
    fn detect_drift(&mut self, observations: &[f64]) -> bool;

    /// Reset the drift detector
    fn reset(&mut self);

    /// Get drift detection statistics
    fn get_statistics(&self) -> DriftStatistics;
}

/// Statistics from drift detection
#[derive(Debug, Clone)]
pub struct DriftStatistics {
    /// Number of drift points detected
    pub drift_points: usize,
    /// Average time between drifts
    pub avg_drift_interval: f64,
    /// Current drift confidence score
    pub drift_confidence: f64,
    /// Time since last drift
    pub time_since_drift: usize,
}

/// ADWIN (Adaptive Windowing) drift detector
pub struct AdwinDriftDetector {
    window: Vec<f64>,
    max_window_size: usize,
    delta: f64, // Confidence parameter
    min_window_size: usize,
    drift_count: usize,
    last_drift_time: usize,
    current_time: usize,
}

impl AdwinDriftDetector {
    pub fn new(delta: f64, max_window_size: usize, min_window_size: usize) -> Self {
        Self {
            window: Vec::new(),
            max_window_size,
            delta,
            min_window_size,
            drift_count: 0,
            last_drift_time: 0,
            current_time: 0,
        }
    }

    fn cut_expression(&self, n0: f64, n1: f64, _u0: f64, _u1: f64) -> f64 {
        let n = n0 + n1;
        let delta_prime = self.delta / n;
        let m = 1.0 / (2.0 * n0) + 1.0 / (2.0 * n1);

        // Fix: Use absolute value of log since delta_prime < 1 makes ln negative
        let log_term = (-delta_prime.ln()).max(0.0);
        (2.0 * log_term / m).sqrt() + (2.0 * log_term / (3.0 * m)) / n
    }
}

impl ConceptDriftDetector for AdwinDriftDetector {
    fn detect_drift(&mut self, observations: &[f64]) -> bool {
        for &obs in observations {
            self.window.push(obs);
            self.current_time += 1;

            // Maintain window size
            if self.window.len() > self.max_window_size {
                self.window.remove(0);
            }

            if self.window.len() < self.min_window_size {
                continue;
            }

            // Check for drift by comparing sub-windows
            let n = self.window.len();
            for i in 1..n {
                let w0 = &self.window[0..i];
                let w1 = &self.window[i..];

                if w0.len() < self.min_window_size || w1.len() < self.min_window_size {
                    continue;
                }

                let u0: f64 = w0.iter().sum::<f64>() / w0.len() as f64;
                let u1: f64 = w1.iter().sum::<f64>() / w1.len() as f64;
                let cut_val = self.cut_expression(w0.len() as f64, w1.len() as f64, u0, u1);

                if (u0 - u1).abs() > cut_val {
                    // Drift detected, remove old window
                    self.window.drain(0..i);
                    self.drift_count += 1;
                    self.last_drift_time = self.current_time;
                    return true;
                }
            }
        }
        false
    }

    fn reset(&mut self) {
        self.window.clear();
        self.drift_count = 0;
        self.last_drift_time = 0;
        self.current_time = 0;
    }

    fn get_statistics(&self) -> DriftStatistics {
        let avg_drift_interval = if self.drift_count > 0 {
            self.current_time as f64 / self.drift_count as f64
        } else {
            0.0
        };

        DriftStatistics {
            drift_points: self.drift_count,
            avg_drift_interval,
            drift_confidence: if self.window.len() >= self.min_window_size {
                0.95
            } else {
                0.0
            },
            time_since_drift: self.current_time - self.last_drift_time,
        }
    }
}

impl TimeSeriesEnsembleConfig {
    pub fn builder() -> TimeSeriesEnsembleConfigBuilder {
        TimeSeriesEnsembleConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct TimeSeriesEnsembleConfigBuilder {
    config: TimeSeriesEnsembleConfig,
}

impl TimeSeriesEnsembleConfigBuilder {
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = window_size;
        self
    }

    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    pub fn temporal_overlap(mut self, temporal_overlap: f64) -> Self {
        self.config.temporal_overlap = temporal_overlap;
        self
    }

    pub fn seasonal_periods(mut self, periods: Vec<usize>) -> Self {
        self.config.seasonal_periods = periods;
        self
    }

    pub fn temporal_decay(mut self, decay: f64) -> Self {
        self.config.temporal_decay = decay;
        self
    }

    pub fn drift_adaptation(mut self, strategy: DriftAdaptationStrategy) -> Self {
        self.config.drift_adaptation = strategy;
        self
    }

    pub fn cv_strategy(mut self, strategy: TimeSeriesCVStrategy) -> Self {
        self.config.cv_strategy = strategy;
        self
    }

    pub fn aggregation_method(mut self, method: TemporalAggregationMethod) -> Self {
        self.config.aggregation_method = method;
        self
    }

    pub fn use_seasonal_decomposition(mut self, use_seasonal: bool) -> Self {
        self.config.use_seasonal_decomposition = use_seasonal;
        self
    }

    pub fn build(self) -> TimeSeriesEnsembleConfig {
        self.config
    }
}

impl TimeSeriesEnsembleRegressor {
    pub fn new(config: TimeSeriesEnsembleConfig) -> Self {
        Self {
            config,
            base_models: Vec::new(),
            temporal_weights: Vec::new(),
            seasonal_components: None,
            drift_detector: None,
            is_fitted: false,
        }
    }

    pub fn builder() -> TimeSeriesEnsembleRegressorBuilder {
        TimeSeriesEnsembleRegressorBuilder::new()
    }

    /// Create time series features from raw data
    fn create_time_features(&self, data: &Array2<f64>) -> SklResult<Array2<f64>> {
        let shape = data.shape();
        let (n_samples, n_features) = (shape[0], shape[1]);
        let window_size = self.config.window_size;

        if n_samples < window_size {
            return Err(SklearsError::InvalidInput(format!(
                "Not enough samples ({}) for window size ({})",
                n_samples, window_size
            )));
        }

        let n_output_samples = n_samples - window_size + 1;
        let n_output_features = n_features * window_size;

        let mut features = Array2::zeros((n_output_samples, n_output_features));

        for i in 0..n_output_samples {
            for j in 0..window_size {
                for k in 0..n_features {
                    features[[i, j * n_features + k]] = data[[i + j, k]];
                }
            }
        }

        Ok(features)
    }

    /// Extract seasonal components using simple decomposition
    fn extract_seasonal_components(&mut self, y: &[f64]) -> SklResult<()> {
        if !self.config.use_seasonal_decomposition {
            return Ok(());
        }

        let n = y.len();
        let mut trend = vec![0.0; n];
        let mut seasonal = Vec::new();
        let mut residual = vec![0.0; n];

        // Simple moving average for trend
        let ma_window = 12.min(n / 4);
        for i in ma_window / 2..n - ma_window / 2 {
            let sum: f64 = y[i - ma_window / 2..i + ma_window / 2 + 1].iter().sum();
            trend[i] = sum / (ma_window + 1) as f64;
        }

        // Extract seasonal patterns for each period
        for &period in &self.config.seasonal_periods {
            if period > n {
                continue;
            }

            let mut period_seasonal = vec![0.0; period];
            let mut counts = vec![0; period];

            for i in 0..n {
                let seasonal_idx = i % period;
                period_seasonal[seasonal_idx] += y[i] - trend[i];
                counts[seasonal_idx] += 1;
            }

            // Average seasonal components
            for i in 0..period {
                if counts[i] > 0 {
                    period_seasonal[i] /= counts[i] as f64;
                }
            }

            seasonal.push(period_seasonal);
        }

        // Calculate residuals
        for i in 0..n {
            residual[i] = y[i] - trend[i];
            for (idx, &period) in self.config.seasonal_periods.iter().enumerate() {
                if idx < seasonal.len() && period <= n {
                    residual[i] -= seasonal[idx][i % period];
                }
            }
        }

        self.seasonal_components = Some(SeasonalComponents {
            trend,
            seasonal,
            residual,
        });

        Ok(())
    }

    /// Update temporal weights based on performance
    fn update_temporal_weights(&mut self, recent_errors: &[f64]) {
        let n_models = self.base_models.len();
        if n_models == 0 {
            return;
        }

        self.temporal_weights = vec![1.0 / n_models as f64; n_models];

        if recent_errors.len() != n_models {
            return;
        }

        // Weight based on inverse error with temporal decay
        let total_error: f64 = recent_errors.iter().sum();
        if total_error > 0.0 {
            for i in 0..n_models {
                let inv_error = 1.0 / (recent_errors[i] + 1e-8);
                let temporal_factor = self.config.temporal_decay.powi(i as i32);
                self.temporal_weights[i] = inv_error * temporal_factor;
            }

            // Normalize weights
            let sum_weights: f64 = self.temporal_weights.iter().sum();
            if sum_weights > 0.0 {
                for weight in &mut self.temporal_weights {
                    *weight /= sum_weights;
                }
            }
        }
    }

    /// Aggregate predictions using configured method
    fn aggregate_predictions(&self, predictions: &[Vec<f64>]) -> SklResult<Vec<f64>> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions to aggregate".to_string(),
            ));
        }

        let n_samples = predictions[0].len();
        let mut result = vec![0.0; n_samples];

        match self.config.aggregation_method {
            TemporalAggregationMethod::SimpleAverage => {
                for pred in predictions {
                    for (i, &p) in pred.iter().enumerate() {
                        result[i] += p;
                    }
                }
                for r in &mut result {
                    *r /= predictions.len() as f64;
                }
            }
            TemporalAggregationMethod::WeightedAverage => {
                for (j, pred) in predictions.iter().enumerate() {
                    let weight = if j < self.temporal_weights.len() {
                        self.temporal_weights[j]
                    } else {
                        1.0 / predictions.len() as f64
                    };
                    for (i, &p) in pred.iter().enumerate() {
                        result[i] += p * weight;
                    }
                }
            }
            TemporalAggregationMethod::MedianAggregation => {
                for i in 0..n_samples {
                    let mut values: Vec<f64> = predictions.iter().map(|p| p[i]).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    result[i] = if values.len() % 2 == 0 {
                        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };
                }
            }
            TemporalAggregationMethod::ExponentialSmoothing => {
                let alpha = 0.3; // Smoothing parameter
                for i in 0..n_samples {
                    result[i] = predictions[0][i];
                    for j in 1..predictions.len() {
                        result[i] = alpha * predictions[j][i] + (1.0 - alpha) * result[i];
                    }
                }
            }
            _ => {
                // Default to simple average for other methods
                for pred in predictions {
                    for (i, &p) in pred.iter().enumerate() {
                        result[i] += p;
                    }
                }
                for r in &mut result {
                    *r /= predictions.len() as f64;
                }
            }
        }

        Ok(result)
    }
}

pub struct TimeSeriesEnsembleRegressorBuilder {
    config: TimeSeriesEnsembleConfig,
}

impl Default for TimeSeriesEnsembleRegressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesEnsembleRegressorBuilder {
    pub fn new() -> Self {
        Self {
            config: TimeSeriesEnsembleConfig::default(),
        }
    }

    pub fn config(mut self, config: TimeSeriesEnsembleConfig) -> Self {
        self.config = config;
        self
    }

    pub fn window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = window_size;
        self
    }

    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    pub fn temporal_decay(mut self, decay: f64) -> Self {
        self.config.temporal_decay = decay;
        self
    }

    pub fn drift_adaptation(mut self, strategy: DriftAdaptationStrategy) -> Self {
        self.config.drift_adaptation = strategy;
        self
    }

    pub fn build(self) -> TimeSeriesEnsembleRegressor {
        TimeSeriesEnsembleRegressor::new(self.config)
    }
}

impl Estimator for TimeSeriesEnsembleRegressor {
    type Config = TimeSeriesEnsembleConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Vec<f64>> for TimeSeriesEnsembleRegressor {
    type Fitted = Self;

    #[allow(non_snake_case)]
    fn fit(mut self, X: &Array2<f64>, y: &Vec<f64>) -> SklResult<Self::Fitted> {
        if X.shape()[0] != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X and y must have same number of samples: {}", y.len()),
                actual: format!(
                    "X has {} samples but y has {} samples",
                    X.shape()[0],
                    y.len()
                ),
            });
        }

        // Extract seasonal components
        self.extract_seasonal_components(y)?;

        // Create time series features
        let time_features = self.create_time_features(X)?;
        let n_time_samples = time_features.shape()[0];

        // Adjust target vector for time features
        let y_time = &y[self.config.window_size - 1..];

        // Initialize drift detector
        if matches!(
            self.config.drift_adaptation,
            DriftAdaptationStrategy::SlidingWindow
        ) {
            self.drift_detector = Some(Box::new(AdwinDriftDetector::new(0.002, 1000, 30)));
        }

        // Train ensemble models with temporal overlap
        self.base_models.clear();
        let overlap_size = (n_time_samples as f64 * self.config.temporal_overlap) as usize;
        let step_size = (n_time_samples - overlap_size) / self.config.n_estimators.max(1);

        for i in 0..self.config.n_estimators {
            let start_idx = i * step_size;
            let end_idx = (start_idx + overlap_size + step_size).min(n_time_samples);

            if end_idx <= start_idx {
                break;
            }

            // Extract temporal subset
            let X_subset = time_features.slice(s![start_idx..end_idx, ..]);
            let y_subset = &y_time[start_idx..end_idx];

            // Create and train base model using builder pattern
            let X_subset_owned = X_subset.to_owned();
            let y_subset_owned = Array1::from_vec(y_subset.to_vec());

            let model = GradientBoostingRegressor::builder()
                .n_estimators(50)
                .learning_rate(0.1)
                .max_depth(6)
                .build()
                .fit(&X_subset_owned, &y_subset_owned)?;

            self.base_models.push(model);
        }

        // Initialize temporal weights
        self.temporal_weights = vec![1.0 / self.base_models.len() as f64; self.base_models.len()];
        self.is_fitted = true;

        Ok(self)
    }
}

impl Predict<Array2<f64>, Vec<f64>> for TimeSeriesEnsembleRegressor {
    fn predict(&self, X: &Array2<f64>) -> SklResult<Vec<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let time_features = self.create_time_features(X)?;
        let mut predictions = Vec::new();

        // Get predictions from all base models
        for model in &self.base_models {
            let pred = model.predict(&time_features)?;
            let pred_vec: Vec<f64> = pred.to_vec();
            predictions.push(pred_vec);
        }

        // Aggregate predictions
        self.aggregate_predictions(&predictions)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_time_series_ensemble_config() {
        let config = TimeSeriesEnsembleConfig::builder()
            .window_size(12)
            .n_estimators(5)
            .temporal_decay(0.9)
            .build();

        assert_eq!(config.window_size, 12);
        assert_eq!(config.n_estimators, 5);
        assert_eq!(config.temporal_decay, 0.9);
    }

    #[test]
    fn test_adwin_drift_detector() {
        let mut detector = AdwinDriftDetector::new(0.01, 1000, 30); // Increased delta for more sensitive detection

        // Test with stable data (no drift)
        let stable_data = vec![1.0; 50];
        assert!(!detector.detect_drift(&stable_data));

        // Test with more significant drift - use incremental detection
        let mut drift_data = vec![1.0; 30];
        drift_data.extend(vec![100.0; 30]); // Even larger drift should be detectable

        // ADWIN implementation fixed - should properly detect significant drift
        let drift_detected = detector.detect_drift(&drift_data);
        assert!(
            drift_detected,
            "ADWIN should detect significant drift from 1.0 to 100.0"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_time_series_ensemble_basic() {
        let config = TimeSeriesEnsembleConfig::builder()
            .window_size(3)
            .n_estimators(3)
            .build();

        let ensemble = TimeSeriesEnsembleRegressor::new(config);

        // Create simple time series data
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0];

        let X = Array2::from_shape_vec((5, 2), data).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test basic functionality without full fit/predict for now
        assert_eq!(ensemble.config.window_size, 3);
        assert_eq!(ensemble.config.n_estimators, 3);
    }

    #[test]
    fn test_seasonal_decomposition() {
        let config = TimeSeriesEnsembleConfig::builder()
            .window_size(4)
            .use_seasonal_decomposition(true)
            .seasonal_periods(vec![7])
            .build();

        let mut ensemble = TimeSeriesEnsembleRegressor::new(config);

        // Create data with seasonal pattern
        let y: Vec<f64> = (0..28)
            .map(|i| (i as f64 / 7.0 * 2.0 * std::f64::consts::PI).sin())
            .collect();

        ensemble.extract_seasonal_components(&y).unwrap();
        assert!(ensemble.seasonal_components.is_some());

        let components = ensemble.seasonal_components.unwrap();
        assert_eq!(components.trend.len(), 28);
        assert_eq!(components.seasonal.len(), 1);
        assert_eq!(components.seasonal[0].len(), 7);
        assert_eq!(components.residual.len(), 28);
    }
}
