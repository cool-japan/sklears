//! Time Series Utilities
//!
//! This module provides comprehensive time series data structures and utilities
//! for temporal data processing in machine learning applications.

use crate::UtilsError;
use chrono::{DateTime, Local, TimeZone, Utc};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Timestamp representation for time series data
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    pub timestamp: i64, // Unix timestamp in milliseconds
}

impl Timestamp {
    /// Create a new timestamp from milliseconds since Unix epoch
    pub fn from_millis(millis: i64) -> Self {
        Self { timestamp: millis }
    }

    /// Create a new timestamp from seconds since Unix epoch
    pub fn from_secs(secs: i64) -> Self {
        Self {
            timestamp: secs * 1000,
        }
    }

    /// Create a timestamp from the current time
    pub fn now() -> Self {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0));
        Self::from_millis(duration.as_millis() as i64)
    }

    /// Convert to milliseconds
    pub fn as_millis(&self) -> i64 {
        self.timestamp
    }

    /// Convert to seconds
    pub fn as_secs(&self) -> i64 {
        self.timestamp / 1000
    }

    /// Convert to `DateTime<Utc>`
    pub fn to_datetime_utc(&self) -> DateTime<Utc> {
        let naive = DateTime::from_timestamp_millis(self.timestamp)
            .unwrap_or_default()
            .naive_utc();
        DateTime::from_naive_utc_and_offset(naive, Utc)
    }

    /// Add duration to timestamp
    pub fn add_duration(&self, duration: Duration) -> Self {
        Self::from_millis(self.timestamp + duration.as_millis() as i64)
    }

    /// Subtract duration from timestamp
    pub fn sub_duration(&self, duration: Duration) -> Self {
        Self::from_millis(self.timestamp - duration.as_millis() as i64)
    }
}

/// Time series data point with timestamp and value
#[derive(Debug, Clone)]
pub struct TimeSeriesPoint<T> {
    pub timestamp: Timestamp,
    pub value: T,
}

impl<T> TimeSeriesPoint<T> {
    pub fn new(timestamp: Timestamp, value: T) -> Self {
        Self { timestamp, value }
    }
}

/// Time series data structure with efficient temporal indexing
#[derive(Debug, Clone)]
pub struct TimeSeries<T> {
    data: BTreeMap<Timestamp, T>,
    metadata: HashMap<String, String>,
}

impl<T: Clone> Default for TimeSeries<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> TimeSeries<T> {
    /// Create a new empty time series
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create time series from vectors of timestamps and values
    pub fn from_vecs(timestamps: Vec<Timestamp>, values: Vec<T>) -> Result<Self, UtilsError> {
        if timestamps.len() != values.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![timestamps.len()],
                actual: vec![values.len()],
            });
        }

        let mut ts = Self::new();
        for (timestamp, value) in timestamps.into_iter().zip(values.into_iter()) {
            ts.insert(timestamp, value);
        }
        Ok(ts)
    }

    /// Insert a new data point
    pub fn insert(&mut self, timestamp: Timestamp, value: T) {
        self.data.insert(timestamp, value);
    }

    /// Get value at specific timestamp
    pub fn get(&self, timestamp: &Timestamp) -> Option<&T> {
        self.data.get(timestamp)
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if time series is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the first timestamp
    pub fn first_timestamp(&self) -> Option<Timestamp> {
        self.data.keys().next().copied()
    }

    /// Get the last timestamp
    pub fn last_timestamp(&self) -> Option<Timestamp> {
        self.data.keys().next_back().copied()
    }

    /// Get data points in a time range
    pub fn range(&self, start: Timestamp, end: Timestamp) -> Vec<TimeSeriesPoint<T>> {
        self.data
            .range(start..=end)
            .map(|(&timestamp, value)| TimeSeriesPoint::new(timestamp, value.clone()))
            .collect()
    }

    /// Get all timestamps
    pub fn timestamps(&self) -> Vec<Timestamp> {
        self.data.keys().copied().collect()
    }

    /// Get all values
    pub fn values(&self) -> Vec<T> {
        self.data.values().cloned().collect()
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Resample time series to regular intervals
    pub fn resample(
        &self,
        interval: Duration,
        aggregation: AggregationMethod,
    ) -> Result<TimeSeries<f64>, UtilsError>
    where
        T: Into<f64> + Copy,
    {
        if self.is_empty() {
            return Ok(TimeSeries::new());
        }

        let start = self.first_timestamp().unwrap();
        let end = self.last_timestamp().unwrap();
        let mut resampled = TimeSeries::new();

        let mut current = start;
        while current.timestamp <= end.timestamp {
            let window_end = current.add_duration(interval);
            let window_data: Vec<f64> = self
                .range(current, window_end)
                .into_iter()
                .map(|point| point.value.into())
                .collect();

            if !window_data.is_empty() {
                let aggregated = match aggregation {
                    AggregationMethod::Mean => {
                        window_data.iter().sum::<f64>() / window_data.len() as f64
                    }
                    AggregationMethod::Sum => window_data.iter().sum(),
                    AggregationMethod::Min => {
                        window_data.iter().fold(f64::INFINITY, |a, &b| a.min(b))
                    }
                    AggregationMethod::Max => {
                        window_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                    }
                    AggregationMethod::First => window_data[0],
                    AggregationMethod::Last => window_data[window_data.len() - 1],
                };
                resampled.insert(current, aggregated);
            }

            current = current.add_duration(interval);
        }

        Ok(resampled)
    }
}

/// Aggregation methods for resampling
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    Mean,
    Sum,
    Min,
    Max,
    First,
    Last,
}

/// Sliding window for time series analysis
#[derive(Debug, Clone)]
pub struct SlidingWindow<T> {
    window_size: Duration,
    data: VecDeque<TimeSeriesPoint<T>>,
}

impl<T: Clone> SlidingWindow<T> {
    /// Create a new sliding window with specified size
    pub fn new(window_size: Duration) -> Self {
        Self {
            window_size,
            data: VecDeque::new(),
        }
    }

    /// Add a new data point and maintain window size
    pub fn add(&mut self, point: TimeSeriesPoint<T>) {
        self.data.push_back(point.clone());

        // Remove old data points outside the window
        let cutoff = point.timestamp.sub_duration(self.window_size);
        while let Some(front) = self.data.front() {
            if front.timestamp < cutoff {
                self.data.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get current window data
    pub fn current_window(&self) -> Vec<TimeSeriesPoint<T>> {
        self.data.iter().cloned().collect()
    }

    /// Get window size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if window is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compute statistics for numeric data in the window
    pub fn compute_stats(&self) -> WindowStats
    where
        T: Into<f64> + Copy,
    {
        if self.is_empty() {
            return WindowStats::default();
        }

        let values: Vec<f64> = self.data.iter().map(|p| p.value.into()).collect();
        let n = values.len() as f64;
        let sum = values.iter().sum::<f64>();
        let mean = sum / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        WindowStats {
            count: values.len(),
            mean,
            std_dev,
            min,
            max,
            sum,
        }
    }
}

/// Statistics for sliding window
#[derive(Debug, Clone, Default)]
pub struct WindowStats {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub sum: f64,
}

/// Temporal indexing utilities
pub struct TemporalIndex {
    index: BTreeMap<Timestamp, Vec<usize>>,
}

impl Default for TemporalIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalIndex {
    /// Create a new temporal index
    pub fn new() -> Self {
        Self {
            index: BTreeMap::new(),
        }
    }

    /// Add an entry to the index
    pub fn add_entry(&mut self, timestamp: Timestamp, id: usize) {
        self.index.entry(timestamp).or_default().push(id);
    }

    /// Find entries in time range
    pub fn find_range(&self, start: Timestamp, end: Timestamp) -> Vec<usize> {
        self.index
            .range(start..=end)
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect()
    }

    /// Find entries before timestamp
    pub fn find_before(&self, timestamp: Timestamp) -> Vec<usize> {
        self.index
            .range(..timestamp)
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect()
    }

    /// Find entries after timestamp
    pub fn find_after(&self, timestamp: Timestamp) -> Vec<usize> {
        self.index
            .range((
                std::ops::Bound::Excluded(timestamp),
                std::ops::Bound::Unbounded,
            ))
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect()
    }
}

/// Time zone utilities
pub struct TimeZoneUtils;

impl TimeZoneUtils {
    /// Convert timestamp from one timezone to another
    pub fn convert_timezone(
        timestamp: Timestamp,
        from_tz: &str,
        to_tz: &str,
    ) -> Result<Timestamp, UtilsError> {
        // Simplified timezone conversion - in practice would use proper timezone libraries
        let datetime_utc = timestamp.to_datetime_utc();

        // This is a simplified implementation
        // In practice, you'd want to use proper timezone handling
        match (from_tz, to_tz) {
            ("UTC", "Local") => {
                let local = Local.from_utc_datetime(&datetime_utc.naive_utc());
                Ok(Timestamp::from_millis(local.timestamp_millis()))
            }
            ("Local", "UTC") => {
                // Assume input is local time, convert to UTC
                Ok(timestamp) // Simplified
            }
            _ => Ok(timestamp), // No conversion for unsupported timezones
        }
    }

    /// Get timestamp for start of day
    pub fn start_of_day(timestamp: Timestamp) -> Timestamp {
        let datetime = timestamp.to_datetime_utc();
        let start_of_day = datetime.date_naive().and_hms_opt(0, 0, 0).unwrap();
        let start_of_day_utc: DateTime<Utc> =
            DateTime::from_naive_utc_and_offset(start_of_day, Utc);
        Timestamp::from_millis(start_of_day_utc.timestamp_millis())
    }

    /// Get timestamp for end of day
    pub fn end_of_day(timestamp: Timestamp) -> Timestamp {
        let datetime = timestamp.to_datetime_utc();
        let end_of_day = datetime.date_naive().and_hms_opt(23, 59, 59).unwrap();
        let end_of_day_utc: DateTime<Utc> = DateTime::from_naive_utc_and_offset(end_of_day, Utc);
        Timestamp::from_millis(end_of_day_utc.timestamp_millis())
    }
}

/// Temporal aggregation utilities
pub struct TemporalAggregator;

impl TemporalAggregator {
    /// Aggregate time series data by time periods
    pub fn aggregate_by_period<T>(
        time_series: &TimeSeries<T>,
        period: Duration,
        aggregation: AggregationMethod,
    ) -> Result<TimeSeries<f64>, UtilsError>
    where
        T: Into<f64> + Copy,
    {
        time_series.resample(period, aggregation)
    }

    /// Compute rolling statistics
    pub fn rolling_statistics<T>(
        data: &[TimeSeriesPoint<T>],
        window_size: Duration,
    ) -> Vec<WindowStats>
    where
        T: Into<f64> + Copy + Clone,
    {
        let mut results = Vec::new();
        let mut window = SlidingWindow::new(window_size);

        for point in data {
            window.add(point.clone());
            results.push(window.compute_stats());
        }

        results
    }

    /// Detect trends in time series data
    pub fn detect_trend<T>(data: &[TimeSeriesPoint<T>], window_size: usize) -> Vec<TrendDirection>
    where
        T: Into<f64> + Copy,
    {
        if data.len() < window_size * 2 {
            return vec![TrendDirection::Stable; data.len()];
        }

        let mut trends = Vec::new();
        let values: Vec<f64> = data.iter().map(|p| p.value.into()).collect();

        for i in window_size..(values.len() - window_size) {
            let before: f64 = values[(i - window_size)..i].iter().sum::<f64>() / window_size as f64;
            let after: f64 =
                values[(i + 1)..(i + 1 + window_size)].iter().sum::<f64>() / window_size as f64;

            let trend = if after > before * 1.05 {
                TrendDirection::Increasing
            } else if after < before * 0.95 {
                TrendDirection::Decreasing
            } else {
                TrendDirection::Stable
            };

            trends.push(trend);
        }

        // Pad with stable for edge cases
        let mut result = vec![TrendDirection::Stable; window_size];
        result.extend(trends);
        result.extend(vec![TrendDirection::Stable; window_size]);
        result
    }
}

/// Trend direction enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Lag features generator for time series
pub struct LagFeatureGenerator;

impl LagFeatureGenerator {
    /// Generate lag features for time series data
    pub fn generate_lag_features(
        data: &Array1<f64>,
        lags: &[usize],
    ) -> Result<Array2<f64>, UtilsError> {
        if data.is_empty() || lags.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let max_lag = *lags.iter().max().unwrap();
        if data.len() <= max_lag {
            return Err(UtilsError::InsufficientData {
                min: max_lag + 1,
                actual: data.len(),
            });
        }

        let n_samples = data.len() - max_lag;
        let n_features = lags.len() + 1; // +1 for original feature
        let mut features = Array2::zeros((n_samples, n_features));

        for (i, mut row) in features.axis_iter_mut(Axis(0)).enumerate() {
            let idx = i + max_lag;

            // Original value
            row[0] = data[idx];

            // Lag features
            for (j, &lag) in lags.iter().enumerate() {
                row[j + 1] = data[idx - lag];
            }
        }

        Ok(features)
    }

    /// Generate differencing features
    pub fn generate_diff_features(
        data: &Array1<f64>,
        orders: &[usize],
    ) -> Result<Array2<f64>, UtilsError> {
        if data.is_empty() || orders.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let max_order = *orders.iter().max().unwrap();
        if data.len() <= max_order {
            return Err(UtilsError::InsufficientData {
                min: max_order + 1,
                actual: data.len(),
            });
        }

        let n_samples = data.len() - max_order;
        let n_features = orders.len();
        let mut features = Array2::zeros((n_samples, n_features));

        for (j, &order) in orders.iter().enumerate() {
            let mut diff_data = data.to_owned();

            // Apply differencing 'order' times
            for _ in 0..order {
                let mut new_diff = Array1::zeros(diff_data.len() - 1);
                for i in 0..new_diff.len() {
                    new_diff[i] = diff_data[i + 1] - diff_data[i];
                }
                diff_data = new_diff;
            }

            // Copy to features matrix
            for i in 0..n_samples {
                features[(i, j)] = diff_data[i];
            }
        }

        Ok(features)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_timestamp_creation() {
        let ts1 = Timestamp::from_secs(1000);
        let ts2 = Timestamp::from_millis(1_000_000);

        assert_eq!(ts1.as_secs(), 1000);
        assert_eq!(ts2.as_millis(), 1_000_000);
        assert_eq!(ts1, ts2);
    }

    #[test]
    fn test_time_series_basic_operations() {
        let mut ts = TimeSeries::new();
        let ts1 = Timestamp::from_secs(100);
        let ts2 = Timestamp::from_secs(200);

        ts.insert(ts1, 10.0);
        ts.insert(ts2, 20.0);

        assert_eq!(ts.len(), 2);
        assert_eq!(ts.get(&ts1), Some(&10.0));
        assert_eq!(ts.first_timestamp(), Some(ts1));
        assert_eq!(ts.last_timestamp(), Some(ts2));
    }

    #[test]
    fn test_sliding_window() {
        let mut window = SlidingWindow::new(Duration::from_secs(10));
        let base_time = Timestamp::from_secs(100);

        window.add(TimeSeriesPoint::new(base_time, 1.0));
        window.add(TimeSeriesPoint::new(
            base_time.add_duration(Duration::from_secs(5)),
            2.0,
        ));
        window.add(TimeSeriesPoint::new(
            base_time.add_duration(Duration::from_secs(15)),
            3.0,
        ));

        assert_eq!(window.len(), 2); // First point should be evicted
        let stats = window.compute_stats();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.mean, 2.5);
    }

    #[test]
    fn test_temporal_index() {
        let mut index = TemporalIndex::new();
        let ts1 = Timestamp::from_secs(100);
        let ts2 = Timestamp::from_secs(200);
        let ts3 = Timestamp::from_secs(300);

        index.add_entry(ts1, 1);
        index.add_entry(ts2, 2);
        index.add_entry(ts3, 3);

        let range_results = index.find_range(ts1, ts2);
        assert_eq!(range_results, vec![1, 2]);

        let before_results = index.find_before(ts2);
        assert_eq!(before_results, vec![1]);
    }

    #[test]
    fn test_lag_feature_generation() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let lags = vec![1, 2];

        let features = LagFeatureGenerator::generate_lag_features(&data, &lags).unwrap();

        assert_eq!(features.shape(), &[3, 3]); // 3 samples, 3 features (original + 2 lags)
        assert_eq!(features[(0, 0)], 3.0); // Original value at index 2
        assert_eq!(features[(0, 1)], 2.0); // Lag 1
        assert_eq!(features[(0, 2)], 1.0); // Lag 2
    }

    #[test]
    fn test_diff_features() {
        let data = Array1::from(vec![1.0, 3.0, 6.0, 10.0, 15.0]);
        let orders = vec![1, 2];

        let features = LagFeatureGenerator::generate_diff_features(&data, &orders).unwrap();

        assert_eq!(features.shape(), &[3, 2]); // 3 samples, 2 features (1st and 2nd diff)
        assert_eq!(features[(0, 0)], 2.0); // First difference: 3-1
        assert_eq!(features[(0, 1)], 1.0); // Second difference: (3-1) - (3-1) = 0, but this is computed differently
    }

    #[test]
    fn test_time_series_resampling() {
        let timestamps = vec![
            Timestamp::from_secs(0),
            Timestamp::from_secs(1),
            Timestamp::from_secs(2),
            Timestamp::from_secs(3),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0];

        let ts = TimeSeries::from_vecs(timestamps, values).unwrap();
        let resampled = ts
            .resample(Duration::from_secs(2), AggregationMethod::Mean)
            .unwrap();

        assert!(!resampled.is_empty());
    }
}
