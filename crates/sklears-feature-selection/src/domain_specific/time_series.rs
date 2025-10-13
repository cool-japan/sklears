//! Time series feature selection module
//!
//! This module provides specialized feature selection algorithms for time series data,
//! including autocorrelation, cross-correlation, and seasonal pattern analysis.

use crate::base::SelectorMixin;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Time series feature selection using autocorrelation and cross-correlation analysis
///
/// This selector analyzes temporal patterns in features to identify the most relevant
/// time series characteristics for prediction. It considers:
/// - Autocorrelation patterns at various lags
/// - Cross-correlation with the target variable
/// - Seasonal patterns (when enabled)
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_feature_selection::domain_specific::time_series::TimeSeriesSelector;
/// use sklears_core::traits::{Fit, Transform};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let selector = TimeSeriesSelector::new()
///     .max_lag(24)
///     .correlation_threshold(0.15)
///     .include_seasonal(true)
///     .seasonal_period(Some(7))
///     .k(Some(10));
///
/// let x = Array2::zeros((100, 20)); // 100 time points, 20 features
/// let y = Array1::zeros(100);       // Target values
///
/// let fitted_selector = selector.fit(&x, &y)?;
/// let transformed_x = fitted_selector.transform(&x)?;
/// ```
#[derive(Debug, Clone)]
pub struct TimeSeriesSelector<State = Untrained> {
    /// Maximum lag to consider for autocorrelation
    max_lag: usize,
    /// Threshold for cross-correlation with target
    correlation_threshold: f64,
    /// Whether to include seasonal features
    include_seasonal: bool,
    /// Seasonal period (e.g., 24 for hourly data with daily seasonality)
    seasonal_period: Option<usize>,
    /// Number of top features to select
    k: Option<usize>,
    state: PhantomData<State>,
    // Trained state
    autocorrelations_: Option<Array2<Float>>,
    cross_correlations_: Option<Array1<Float>>,
    seasonal_importances_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
}

impl Default for TimeSeriesSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesSelector<Untrained> {
    /// Create a new time series feature selector with default parameters
    ///
    /// Default configuration:
    /// - `max_lag`: 10
    /// - `correlation_threshold`: 0.1
    /// - `include_seasonal`: false
    /// - `seasonal_period`: None
    /// - `k`: None (use threshold-based selection)
    pub fn new() -> Self {
        Self {
            max_lag: 10,
            correlation_threshold: 0.1,
            include_seasonal: false,
            seasonal_period: None,
            k: None,
            state: PhantomData,
            autocorrelations_: None,
            cross_correlations_: None,
            seasonal_importances_: None,
            selected_features_: None,
        }
    }

    /// Set the maximum lag to consider for autocorrelation analysis
    ///
    /// Higher lags capture longer-term temporal dependencies but require
    /// more samples for reliable estimation.
    pub fn max_lag(mut self, max_lag: usize) -> Self {
        self.max_lag = max_lag;
        self
    }

    /// Set the correlation threshold for feature selection
    ///
    /// Features with cross-correlation below this threshold will be filtered out
    /// (when not using k-based selection).
    pub fn correlation_threshold(mut self, threshold: f64) -> Self {
        self.correlation_threshold = threshold;
        self
    }

    /// Enable or disable seasonal pattern analysis
    ///
    /// When enabled, the selector will compute seasonal autocorrelations
    /// and include them in the feature scoring.
    pub fn include_seasonal(mut self, include_seasonal: bool) -> Self {
        self.include_seasonal = include_seasonal;
        self
    }

    /// Set the seasonal period for seasonal pattern analysis
    ///
    /// Common values:
    /// - 7 for daily data with weekly seasonality
    /// - 24 for hourly data with daily seasonality
    /// - 12 for monthly data with yearly seasonality
    pub fn seasonal_period(mut self, period: Option<usize>) -> Self {
        self.seasonal_period = period;
        self
    }

    /// Set the number of top features to select
    ///
    /// When set to `Some(k)`, selects the top k features by score.
    /// When set to `None`, uses threshold-based selection.
    pub fn k(mut self, k: Option<usize>) -> Self {
        self.k = k;
        self
    }
}

impl Estimator for TimeSeriesSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for TimeSeriesSelector<Untrained> {
    type Fitted = TimeSeriesSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();
        if n_samples <= self.max_lag {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than max_lag".to_string(),
            ));
        }

        // Compute autocorrelations for each feature
        let mut autocorrelations = Array2::zeros((n_features, self.max_lag));
        for (i, feature) in x.axis_iter(Axis(1)).enumerate() {
            for lag in 1..=self.max_lag {
                if n_samples > lag {
                    let autocorr = compute_autocorrelation(&feature, lag);
                    autocorrelations[[i, lag - 1]] = autocorr;
                }
            }
        }

        // Compute cross-correlations with target
        let mut cross_correlations = Array1::zeros(n_features);
        for (i, feature) in x.axis_iter(Axis(1)).enumerate() {
            cross_correlations[i] = compute_cross_correlation(&feature, y, 0);
        }

        // Compute seasonal importances if requested
        let seasonal_importances = if self.include_seasonal && self.seasonal_period.is_some() {
            let period = self.seasonal_period.unwrap();
            let mut importances = Array1::zeros(n_features);
            for (i, feature) in x.axis_iter(Axis(1)).enumerate() {
                importances[i] = compute_seasonal_importance(&feature, period);
            }
            Some(importances)
        } else {
            None
        };

        // Select features based on combined score
        let feature_scores =
            self.compute_feature_scores(&cross_correlations, &seasonal_importances);
        let selected_features = self.select_features_from_scores(&feature_scores, n_features);

        Ok(TimeSeriesSelector {
            max_lag: self.max_lag,
            correlation_threshold: self.correlation_threshold,
            include_seasonal: self.include_seasonal,
            seasonal_period: self.seasonal_period,
            k: self.k,
            state: PhantomData,
            autocorrelations_: Some(autocorrelations),
            cross_correlations_: Some(cross_correlations),
            seasonal_importances_: seasonal_importances,
            selected_features_: Some(selected_features),
        })
    }
}

impl TimeSeriesSelector<Untrained> {
    fn compute_feature_scores(
        &self,
        cross_correlations: &Array1<Float>,
        seasonal_importances: &Option<Array1<Float>>,
    ) -> Array1<Float> {
        let mut scores = cross_correlations.abs();

        if let Some(seasonal) = seasonal_importances {
            // Combine cross-correlation and seasonal importance
            for i in 0..scores.len() {
                scores[i] = 0.7 * scores[i] + 0.3 * seasonal[i];
            }
        }

        scores
    }

    fn select_features_from_scores(&self, scores: &Array1<Float>, n_features: usize) -> Vec<usize> {
        let mut feature_indices: Vec<(usize, Float)> = scores
            .indexed_iter()
            .map(|(i, &score)| (i, score))
            .collect();

        // Sort by score descending
        feature_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Apply selection criteria
        let mut selected = Vec::new();

        if let Some(k) = self.k {
            // Select top k features
            selected.extend(
                feature_indices
                    .iter()
                    .take(k.min(n_features))
                    .map(|(i, _)| *i),
            );
        } else {
            // Select features above threshold
            selected.extend(
                feature_indices
                    .iter()
                    .filter(|(_, score)| *score >= self.correlation_threshold)
                    .map(|(i, _)| *i),
            );
        }

        selected.sort();
        selected
    }
}

impl Transform<Array2<Float>> for TimeSeriesSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for TimeSeriesSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.autocorrelations_.as_ref().unwrap().nrows();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            support[idx] = true;
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl TimeSeriesSelector<Trained> {
    /// Get the autocorrelation coefficients for all features and lags
    ///
    /// Returns a matrix where rows correspond to features and columns to lags.
    pub fn autocorrelations(&self) -> &Array2<Float> {
        self.autocorrelations_.as_ref().unwrap()
    }

    /// Get the cross-correlation coefficients with the target
    ///
    /// Returns an array where each element is the cross-correlation
    /// between the corresponding feature and the target variable.
    pub fn cross_correlations(&self) -> &Array1<Float> {
        self.cross_correlations_.as_ref().unwrap()
    }

    /// Get the seasonal importance scores (if seasonal analysis was enabled)
    ///
    /// Returns `None` if seasonal analysis was not enabled during fitting.
    pub fn seasonal_importances(&self) -> Option<&Array1<Float>> {
        self.seasonal_importances_.as_ref()
    }

    /// Get the indices of selected features
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_selected(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Compute a lagged mean-reversion correlation for a time series
///
/// Rather than the classical autocorrelation (which primarily measures
/// persistence), this helper focuses on capturing *mean-reversion*
/// behaviour. For each lag `k`, we compute the Pearson correlation between
/// the series values at time *t* and the forward differences
/// `x_{t+k} - x_t`. A positive value indicates momentum (high values tend
/// to be followed by further increases), while a negative value reflects
/// mean reversion (high values are typically followed by decreases).
fn compute_autocorrelation(series: &ArrayView1<Float>, lag: usize) -> Float {
    let n = series.len();
    if n <= lag {
        return 0.0;
    }

    let window_length = n - lag;
    let mut base_mean = 0.0;
    let mut diff_mean = 0.0;

    for i in 0..window_length {
        let base = series[i];
        let forward = series[i + lag];
        base_mean += base;
        diff_mean += forward - base;
    }

    let window_length_f = window_length as Float;
    base_mean /= window_length_f;
    diff_mean /= window_length_f;

    let mut numerator = 0.0;
    let mut base_var = 0.0;
    let mut diff_var = 0.0;

    for i in 0..window_length {
        let base = series[i] - base_mean;
        let diff = (series[i + lag] - series[i]) - diff_mean;
        numerator += base * diff;
        base_var += base * base;
        diff_var += diff * diff;
    }

    let denominator = (base_var * diff_var).sqrt();
    if denominator.abs() < Float::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute the cross-correlation between two time series at a given lag
///
/// Uses the Pearson correlation formula between x(t) and y(t+lag).
/// Positive lag means y leads x, negative lag means x leads y.
fn compute_cross_correlation(x: &ArrayView1<Float>, y: &Array1<Float>, lag: usize) -> Float {
    let n = x.len().min(y.len());
    if n <= lag {
        return 0.0;
    }

    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..(n - lag) {
        let x_i = x[i] - x_mean;
        let y_i = y[i + lag] - y_mean;
        numerator += x_i * y_i;
        x_var += x_i * x_i;
        y_var += y_i * y_i;
    }

    let denominator = (x_var * y_var).sqrt();
    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute the seasonal importance of a time series
///
/// Measures the strength of seasonal patterns by computing the
/// autocorrelation at the seasonal lag (period).
fn compute_classical_autocorrelation(series: &ArrayView1<Float>, lag: usize) -> Float {
    let n = series.len();
    if n <= lag {
        return 0.0;
    }

    let mean = series.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut variance = 0.0;

    for i in 0..(n - lag) {
        let a = series[i] - mean;
        let b = series[i + lag] - mean;
        numerator += a * b;
    }

    for value in series.iter() {
        let centered = *value - mean;
        variance += centered * centered;
    }

    if variance.abs() < Float::EPSILON {
        0.0
    } else {
        numerator / variance
    }
}

fn compute_seasonal_importance(series: &ArrayView1<Float>, period: usize) -> Float {
    let n = series.len();
    if n < 2 * period {
        return 0.0;
    }

    // Compute seasonal autocorrelation
    compute_classical_autocorrelation(series, period).abs()
}

/// Create a new time series feature selector
pub fn create_time_series_selector() -> TimeSeriesSelector<Untrained> {
    TimeSeriesSelector::new()
}

/// Create a time series selector optimized for high-frequency data
///
/// Suitable for minute-level or second-level data where short-term
/// patterns are more important than long-term seasonality.
pub fn create_high_frequency_selector() -> TimeSeriesSelector<Untrained> {
    TimeSeriesSelector::new()
        .max_lag(60) // Look back 1 hour for minute data
        .correlation_threshold(0.05)
        .include_seasonal(false)
}

/// Create a time series selector optimized for daily data
///
/// Suitable for daily observations with weekly and potentially
/// longer-term seasonal patterns.
pub fn create_daily_selector() -> TimeSeriesSelector<Untrained> {
    TimeSeriesSelector::new()
        .max_lag(30) // Look back 1 month
        .correlation_threshold(0.1)
        .include_seasonal(true)
        .seasonal_period(Some(7)) // Weekly seasonality
}

/// Create a time series selector optimized for hourly data
///
/// Suitable for hourly observations with daily and weekly
/// seasonal patterns.
pub fn create_hourly_selector() -> TimeSeriesSelector<Untrained> {
    TimeSeriesSelector::new()
        .max_lag(168) // Look back 1 week (168 hours)
        .correlation_threshold(0.08)
        .include_seasonal(true)
        .seasonal_period(Some(24)) // Daily seasonality
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_autocorrelation_computation() {
        let series = array![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let autocorr_1 = compute_autocorrelation(&series.view(), 1);

        // Should have negative autocorrelation at lag 1 due to the pattern
        assert!(autocorr_1 < 0.0);
    }

    #[test]
    fn test_cross_correlation_computation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let cross_corr = compute_cross_correlation(&x.view(), &y, 0);

        // Perfect positive correlation
        assert!((cross_corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_importance() {
        let series = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let seasonal_imp = compute_seasonal_importance(&series.view(), 2);

        // Should detect strong seasonality with period 2
        assert!(seasonal_imp > 0.5);
    }

    #[test]
    fn test_time_series_selector_basic() {
        let selector = TimeSeriesSelector::new().max_lag(2).k(Some(2));

        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0, 5.0, 10.0, 15.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = selector.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_features_selected(), 2);

        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed.ncols(), 2);
    }
}
