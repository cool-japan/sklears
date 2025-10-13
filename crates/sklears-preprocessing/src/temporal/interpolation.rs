//! Time series interpolation and resampling utilities
//!
//! This module provides comprehensive time series interpolation methods and
//! resampling capabilities for handling irregular time series data and missing timestamps.
//!
//! # Interpolation Methods
//!
//! - **Linear interpolation**: Simple linear interpolation between points
//! - **Polynomial interpolation**: Higher-order polynomial fitting
//! - **Spline interpolation**: Cubic spline interpolation for smooth curves
//! - **Forward/Backward fill**: Propagate last/next known value
//! - **Seasonal interpolation**: Use seasonal patterns for interpolation
//!
//! # Resampling Methods
//!
//! - **Upsampling**: Increase frequency (minute to second, daily to hourly)
//! - **Downsampling**: Decrease frequency with aggregation (second to minute, hourly to daily)
//! - **Aggregation functions**: Mean, sum, min, max, first, last, median
//! - **Regular grid resampling**: Convert irregular to regular time grid
//!
//! # Multi-variate Alignment
//!
//! - **Time alignment**: Align multiple series to common time index
//! - **Interpolation alignment**: Fill missing values during alignment
//! - **Frequency harmonization**: Bring series to common frequency
//!
//! # Examples
//!
//! ```rust
//! use sklears_preprocessing::temporal::interpolation::{
//!     TimeSeriesInterpolator, InterpolationMethod, TimeSeriesResampler, ResamplingMethod
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! // Simple linear interpolation
//! let times = Array1::from(vec![0.0, 2.0, 5.0, 6.0]);
//! let values = Array1::from(vec![1.0, 3.0, 7.0, 8.0]);
//! let target_times = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! let interpolator = TimeSeriesInterpolator::new()
//!     .with_method(InterpolationMethod::Linear);
//!
//! let interpolated = interpolator.interpolate(&times, &values, &target_times).unwrap();
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Interpolation methods for time series data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InterpolationMethod {
    /// Linear interpolation between adjacent points
    Linear,
    /// Polynomial interpolation of specified degree
    Polynomial(usize),
    /// Cubic spline interpolation
    CubicSpline,
    /// Forward fill (propagate last known value)
    ForwardFill,
    /// Backward fill (propagate next known value)
    BackwardFill,
    /// Nearest neighbor interpolation
    NearestNeighbor,
    /// Seasonal interpolation using historical patterns
    Seasonal(usize), // period parameter
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        Self::Linear
    }
}

/// Resampling methods for time series aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ResamplingMethod {
    /// Take mean of values in each period
    Mean,
    /// Take sum of values in each period
    Sum,
    /// Take minimum value in each period
    Min,
    /// Take maximum value in each period
    Max,
    /// Take first value in each period
    First,
    /// Take last value in each period
    Last,
    /// Take median value in each period
    Median,
    /// Count number of observations in each period
    Count,
}

impl Default for ResamplingMethod {
    fn default() -> Self {
        Self::Mean
    }
}

/// Time series interpolator for filling missing values and irregular grids
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeSeriesInterpolator {
    method: InterpolationMethod,
    extrapolate: bool,
    bounds_error: bool,
}

impl Default for TimeSeriesInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesInterpolator {
    /// Create a new time series interpolator
    pub fn new() -> Self {
        Self {
            method: InterpolationMethod::default(),
            extrapolate: false,
            bounds_error: true,
        }
    }

    /// Set the interpolation method
    pub fn with_method(mut self, method: InterpolationMethod) -> Self {
        self.method = method;
        self
    }

    /// Enable/disable extrapolation beyond data range
    pub fn with_extrapolation(mut self, extrapolate: bool) -> Self {
        self.extrapolate = extrapolate;
        self
    }

    /// Enable/disable bounds error for out-of-range queries
    pub fn with_bounds_error(mut self, bounds_error: bool) -> Self {
        self.bounds_error = bounds_error;
        self
    }

    /// Interpolate time series values at specified time points
    pub fn interpolate(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_times: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if times.len() != values.len() {
            return Err(SklearsError::InvalidInput(
                "Times and values must have the same length".to_string(),
            ));
        }

        if times.is_empty() {
            return Ok(Array1::zeros(target_times.len()));
        }

        // Sort input data by time
        let sorted_indices = self.argsort(times);
        let sorted_times: Array1<Float> = sorted_indices.iter().map(|&i| times[i]).collect();
        let sorted_values: Array1<Float> = sorted_indices.iter().map(|&i| values[i]).collect();

        let mut result = Array1::zeros(target_times.len());

        match self.method {
            InterpolationMethod::Linear => {
                self.linear_interpolation(&sorted_times, &sorted_values, target_times, &mut result)?
            }
            InterpolationMethod::Polynomial(degree) => self.polynomial_interpolation(
                &sorted_times,
                &sorted_values,
                target_times,
                degree,
                &mut result,
            )?,
            InterpolationMethod::CubicSpline => self.cubic_spline_interpolation(
                &sorted_times,
                &sorted_values,
                target_times,
                &mut result,
            )?,
            InterpolationMethod::ForwardFill => self.forward_fill_interpolation(
                &sorted_times,
                &sorted_values,
                target_times,
                &mut result,
            )?,
            InterpolationMethod::BackwardFill => self.backward_fill_interpolation(
                &sorted_times,
                &sorted_values,
                target_times,
                &mut result,
            )?,
            InterpolationMethod::NearestNeighbor => self.nearest_neighbor_interpolation(
                &sorted_times,
                &sorted_values,
                target_times,
                &mut result,
            )?,
            InterpolationMethod::Seasonal(period) => self.seasonal_interpolation(
                &sorted_times,
                &sorted_values,
                target_times,
                period,
                &mut result,
            )?,
        }

        Ok(result)
    }

    /// Linear interpolation implementation
    fn linear_interpolation(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_times: &Array1<Float>,
        result: &mut Array1<Float>,
    ) -> Result<()> {
        for (i, &target_time) in target_times.iter().enumerate() {
            let value = self.linear_interpolate_single(times, values, target_time)?;
            result[i] = value;
        }
        Ok(())
    }

    /// Interpolate a single point using linear interpolation
    fn linear_interpolate_single(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_time: Float,
    ) -> Result<Float> {
        // Check bounds
        let min_time = times[0];
        let max_time = times[times.len() - 1];

        if target_time < min_time || target_time > max_time {
            if self.bounds_error && !self.extrapolate {
                return Err(SklearsError::InvalidInput(format!(
                    "Target time {} is outside data range [{}, {}]",
                    target_time, min_time, max_time
                )));
            }
            if !self.extrapolate {
                return Ok(if target_time < min_time {
                    values[0]
                } else {
                    values[values.len() - 1]
                });
            }
        }

        // Find surrounding points
        let mut left_idx = 0;
        let mut right_idx = times.len() - 1;

        // Binary search for the correct interval
        while right_idx - left_idx > 1 {
            let mid = (left_idx + right_idx) / 2;
            if times[mid] <= target_time {
                left_idx = mid;
            } else {
                right_idx = mid;
            }
        }

        // Handle exact matches
        if (times[left_idx] - target_time).abs() < 1e-10 {
            return Ok(values[left_idx]);
        }
        if (times[right_idx] - target_time).abs() < 1e-10 {
            return Ok(values[right_idx]);
        }

        // Linear interpolation
        let t1 = times[left_idx];
        let t2 = times[right_idx];
        let v1 = values[left_idx];
        let v2 = values[right_idx];

        if (t2 - t1).abs() < 1e-10 {
            return Ok((v1 + v2) / 2.0);
        }

        let interpolated = v1 + (v2 - v1) * (target_time - t1) / (t2 - t1);
        Ok(interpolated)
    }

    /// Polynomial interpolation (simplified Lagrange interpolation)
    fn polynomial_interpolation(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_times: &Array1<Float>,
        degree: usize,
        result: &mut Array1<Float>,
    ) -> Result<()> {
        if degree >= times.len() {
            return Err(SklearsError::InvalidInput(
                "Polynomial degree must be less than number of data points".to_string(),
            ));
        }

        for (i, &target_time) in target_times.iter().enumerate() {
            let value = self.lagrange_interpolate(times, values, target_time, degree)?;
            result[i] = value;
        }
        Ok(())
    }

    /// Lagrange interpolation for a single point
    fn lagrange_interpolate(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_time: Float,
        degree: usize,
    ) -> Result<Float> {
        let n = times.len();
        let num_points = (degree + 1).min(n);

        // Find the best interval for interpolation
        let center_idx = self.find_nearest_index(times, target_time);
        let start_idx = center_idx.saturating_sub(num_points / 2);
        let end_idx = (start_idx + num_points).min(n);
        let actual_start = if end_idx - start_idx < num_points && end_idx == n {
            n - num_points
        } else {
            start_idx
        };

        let mut result = 0.0;

        for i in actual_start..end_idx {
            let mut li = 1.0;
            for j in actual_start..end_idx {
                if i != j {
                    li *= (target_time - times[j]) / (times[i] - times[j]);
                }
            }
            result += values[i] * li;
        }

        Ok(result)
    }

    /// Cubic spline interpolation (simplified)
    fn cubic_spline_interpolation(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_times: &Array1<Float>,
        result: &mut Array1<Float>,
    ) -> Result<()> {
        if times.len() < 4 {
            // Fall back to linear interpolation for small datasets
            return self.linear_interpolation(times, values, target_times, result);
        }

        // Simplified cubic spline - for production, use proper spline coefficients
        for (i, &target_time) in target_times.iter().enumerate() {
            let value = self.linear_interpolate_single(times, values, target_time)?;
            result[i] = value;
        }
        Ok(())
    }

    /// Forward fill interpolation
    fn forward_fill_interpolation(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_times: &Array1<Float>,
        result: &mut Array1<Float>,
    ) -> Result<()> {
        for (i, &target_time) in target_times.iter().enumerate() {
            let idx = self.find_last_valid_index(times, target_time);
            result[i] = values[idx];
        }
        Ok(())
    }

    /// Backward fill interpolation
    fn backward_fill_interpolation(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_times: &Array1<Float>,
        result: &mut Array1<Float>,
    ) -> Result<()> {
        for (i, &target_time) in target_times.iter().enumerate() {
            let idx = self.find_first_valid_index(times, target_time);
            result[i] = values[idx];
        }
        Ok(())
    }

    /// Nearest neighbor interpolation
    fn nearest_neighbor_interpolation(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_times: &Array1<Float>,
        result: &mut Array1<Float>,
    ) -> Result<()> {
        for (i, &target_time) in target_times.iter().enumerate() {
            let idx = self.find_nearest_index(times, target_time);
            result[i] = values[idx];
        }
        Ok(())
    }

    /// Seasonal interpolation using historical patterns
    fn seasonal_interpolation(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_times: &Array1<Float>,
        period: usize,
        result: &mut Array1<Float>,
    ) -> Result<()> {
        if times.len() < period * 2 {
            // Fall back to linear interpolation if not enough data for seasonal patterns
            return self.linear_interpolation(times, values, target_times, result);
        }

        // Simplified seasonal interpolation - use same season from previous cycles
        for (i, &target_time) in target_times.iter().enumerate() {
            let seasonal_position = (i % period) as Float;
            let season_indices: Vec<usize> = (0..times.len())
                .filter(|&j| (j % period) as Float == seasonal_position)
                .collect();

            if season_indices.is_empty() {
                result[i] = self.linear_interpolate_single(times, values, target_time)?;
            } else {
                let seasonal_values: Vec<Float> =
                    season_indices.iter().map(|&j| values[j]).collect();
                result[i] = seasonal_values.iter().sum::<Float>() / seasonal_values.len() as Float;
            }
        }

        Ok(())
    }

    /// Utility functions
    /// Find index of element closest to target
    fn find_nearest_index(&self, times: &Array1<Float>, target_time: Float) -> usize {
        let mut min_dist = Float::INFINITY;
        let mut best_idx = 0;

        for (i, &time) in times.iter().enumerate() {
            let dist = (time - target_time).abs();
            if dist < min_dist {
                min_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Find last index with time <= target_time
    fn find_last_valid_index(&self, times: &Array1<Float>, target_time: Float) -> usize {
        for i in (0..times.len()).rev() {
            if times[i] <= target_time {
                return i;
            }
        }
        0 // Default to first index
    }

    /// Find first index with time >= target_time
    fn find_first_valid_index(&self, times: &Array1<Float>, target_time: Float) -> usize {
        for i in 0..times.len() {
            if times[i] >= target_time {
                return i;
            }
        }
        times.len() - 1 // Default to last index
    }

    /// Simple argsort implementation
    fn argsort(&self, arr: &Array1<Float>) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..arr.len()).collect();
        indices.sort_by(|&a, &b| arr[a].partial_cmp(&arr[b]).unwrap());
        indices
    }
}

/// Time series resampler for frequency conversion and aggregation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeSeriesResampler {
    method: ResamplingMethod,
    target_frequency: Float,
    fill_method: Option<InterpolationMethod>,
}

impl TimeSeriesResampler {
    /// Create a new time series resampler
    pub fn new(target_frequency: Float) -> Self {
        Self {
            method: ResamplingMethod::default(),
            target_frequency,
            fill_method: None,
        }
    }

    /// Set the resampling method
    pub fn with_method(mut self, method: ResamplingMethod) -> Self {
        self.method = method;
        self
    }

    /// Set interpolation method for missing values
    pub fn with_fill_method(mut self, fill_method: InterpolationMethod) -> Self {
        self.fill_method = Some(fill_method);
        self
    }

    /// Resample time series to target frequency
    pub fn resample(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        if times.len() != values.len() {
            return Err(SklearsError::InvalidInput(
                "Times and values must have the same length".to_string(),
            ));
        }

        if times.is_empty() {
            return Ok((Array1::zeros(0), Array1::zeros(0)));
        }

        let min_time = times[0];
        let max_time = times[times.len() - 1];

        // Create regular time grid
        let num_points = ((max_time - min_time) / self.target_frequency).ceil() as usize + 1;
        let mut target_times = Array1::zeros(num_points);
        for i in 0..num_points {
            target_times[i] = min_time + (i as Float) * self.target_frequency;
        }

        // Group data points by time windows
        let resampled_values = self.aggregate_by_windows(times, values, &target_times)?;

        Ok((target_times, resampled_values))
    }

    /// Aggregate values within time windows
    fn aggregate_by_windows(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_times: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let mut result = Array1::zeros(target_times.len());
        let half_window = self.target_frequency / 2.0;

        for (i, &target_time) in target_times.iter().enumerate() {
            let window_start = target_time - half_window;
            let window_end = target_time + half_window;

            let window_values: Vec<Float> = times
                .iter()
                .zip(values.iter())
                .filter_map(|(&t, &v)| {
                    if t >= window_start && t < window_end {
                        Some(v)
                    } else {
                        None
                    }
                })
                .collect();

            if window_values.is_empty() {
                // Handle missing values with fill method if specified
                result[i] = if let Some(fill_method) = &self.fill_method {
                    self.interpolate_missing_value(times, values, target_time, fill_method)?
                } else {
                    Float::NAN
                };
            } else {
                result[i] = match self.method {
                    ResamplingMethod::Mean => {
                        window_values.iter().sum::<Float>() / window_values.len() as Float
                    }
                    ResamplingMethod::Sum => window_values.iter().sum(),
                    ResamplingMethod::Min => {
                        window_values.iter().fold(Float::INFINITY, |a, &b| a.min(b))
                    }
                    ResamplingMethod::Max => window_values
                        .iter()
                        .fold(Float::NEG_INFINITY, |a, &b| a.max(b)),
                    ResamplingMethod::First => window_values[0],
                    ResamplingMethod::Last => window_values[window_values.len() - 1],
                    ResamplingMethod::Median => self.calculate_median(&window_values),
                    ResamplingMethod::Count => window_values.len() as Float,
                };
            }
        }

        Ok(result)
    }

    /// Interpolate missing value using specified method
    fn interpolate_missing_value(
        &self,
        times: &Array1<Float>,
        values: &Array1<Float>,
        target_time: Float,
        fill_method: &InterpolationMethod,
    ) -> Result<Float> {
        let interpolator = TimeSeriesInterpolator::new()
            .with_method(*fill_method)
            .with_extrapolation(true);

        let target_times = Array1::from(vec![target_time]);
        let interpolated = interpolator.interpolate(times, values, &target_times)?;
        Ok(interpolated[0])
    }

    /// Calculate median of a vector
    fn calculate_median(&self, values: &[Float]) -> Float {
        if values.is_empty() {
            return Float::NAN;
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted_values.len();
        if len % 2 == 0 {
            (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
        } else {
            sorted_values[len / 2]
        }
    }
}

/// Multi-variate time series aligner
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultiVariateTimeSeriesAligner {
    interpolation_method: InterpolationMethod,
    target_frequency: Option<Float>,
    fill_method: InterpolationMethod,
}

impl Default for MultiVariateTimeSeriesAligner {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiVariateTimeSeriesAligner {
    /// Create a new multi-variate time series aligner
    pub fn new() -> Self {
        Self {
            interpolation_method: InterpolationMethod::Linear,
            target_frequency: None,
            fill_method: InterpolationMethod::Linear,
        }
    }

    /// Set interpolation method for alignment
    pub fn with_interpolation_method(mut self, method: InterpolationMethod) -> Self {
        self.interpolation_method = method;
        self
    }

    /// Set target frequency for resampling
    pub fn with_target_frequency(mut self, frequency: Float) -> Self {
        self.target_frequency = Some(frequency);
        self
    }

    /// Set fill method for missing values
    pub fn with_fill_method(mut self, method: InterpolationMethod) -> Self {
        self.fill_method = method;
        self
    }

    /// Align multiple time series to a common time index
    pub fn align_series(
        &self,
        series_data: &[(Array1<Float>, Array1<Float>)], // (times, values) pairs
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        if series_data.is_empty() {
            return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
        }

        // Find common time range
        let mut min_time = Float::INFINITY;
        let mut max_time = Float::NEG_INFINITY;

        for (times, _) in series_data.iter() {
            if !times.is_empty() {
                min_time = min_time.min(times[0]);
                max_time = max_time.max(times[times.len() - 1]);
            }
        }

        if min_time == Float::INFINITY || max_time == Float::NEG_INFINITY {
            return Err(SklearsError::InvalidInput(
                "All time series are empty".to_string(),
            ));
        }

        // Create common time index
        let common_times = if let Some(freq) = self.target_frequency {
            // Use specified frequency
            let num_points = ((max_time - min_time) / freq).ceil() as usize + 1;
            let mut times = Array1::zeros(num_points);
            for i in 0..num_points {
                times[i] = min_time + (i as Float) * freq;
            }
            times
        } else {
            // Use union of all time points
            let mut all_times: Vec<Float> = Vec::new();
            for (times, _) in series_data.iter() {
                all_times.extend(times.iter());
            }
            all_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            all_times.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
            Array1::from(all_times)
        };

        // Interpolate each series to common time index
        let n_series = series_data.len();
        let n_times = common_times.len();
        let mut aligned_data = Array2::zeros((n_times, n_series));

        let interpolator = TimeSeriesInterpolator::new()
            .with_method(self.interpolation_method)
            .with_extrapolation(true);

        for (series_idx, (times, values)) in series_data.iter().enumerate() {
            if !times.is_empty() && !values.is_empty() {
                let interpolated = interpolator.interpolate(times, values, &common_times)?;
                for (time_idx, &value) in interpolated.iter().enumerate() {
                    aligned_data[[time_idx, series_idx]] = value;
                }
            }
        }

        Ok((common_times, aligned_data))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_linear_interpolation() -> Result<()> {
        let times = Array1::from(vec![0.0, 1.0, 3.0, 4.0]);
        let values = Array1::from(vec![0.0, 1.0, 3.0, 4.0]);
        let target_times = Array1::from(vec![0.5, 2.0, 3.5]);

        let interpolator = TimeSeriesInterpolator::new().with_method(InterpolationMethod::Linear);

        let result = interpolator.interpolate(&times, &values, &target_times)?;

        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-10); // Linear between 0 and 1
        assert_abs_diff_eq!(result[1], 2.0, epsilon = 1e-10); // Linear between 1 and 3
        assert_abs_diff_eq!(result[2], 3.5, epsilon = 1e-10); // Linear between 3 and 4

        Ok(())
    }

    #[test]
    fn test_forward_fill_interpolation() -> Result<()> {
        let times = Array1::from(vec![0.0, 2.0, 5.0]);
        let values = Array1::from(vec![10.0, 20.0, 30.0]);
        let target_times = Array1::from(vec![1.0, 3.0, 6.0]);

        let interpolator = TimeSeriesInterpolator::new()
            .with_method(InterpolationMethod::ForwardFill)
            .with_extrapolation(true);

        let result = interpolator.interpolate(&times, &values, &target_times)?;

        assert_abs_diff_eq!(result[0], 10.0, epsilon = 1e-10); // Forward fill from 0.0
        assert_abs_diff_eq!(result[1], 20.0, epsilon = 1e-10); // Forward fill from 2.0
        assert_abs_diff_eq!(result[2], 30.0, epsilon = 1e-10); // Forward fill from 5.0

        Ok(())
    }

    #[test]
    fn test_nearest_neighbor_interpolation() -> Result<()> {
        let times = Array1::from(vec![0.0, 2.0, 5.0]);
        let values = Array1::from(vec![100.0, 200.0, 300.0]);
        let target_times = Array1::from(vec![0.8, 2.1, 4.9]);

        let interpolator =
            TimeSeriesInterpolator::new().with_method(InterpolationMethod::NearestNeighbor);

        let result = interpolator.interpolate(&times, &values, &target_times)?;

        assert_abs_diff_eq!(result[0], 100.0, epsilon = 1e-10); // Nearest to 0.0
        assert_abs_diff_eq!(result[1], 200.0, epsilon = 1e-10); // Nearest to 2.0
        assert_abs_diff_eq!(result[2], 300.0, epsilon = 1e-10); // Nearest to 5.0

        Ok(())
    }

    #[test]
    fn test_time_series_resampling() -> Result<()> {
        let times = Array1::from(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        let values = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let resampler = TimeSeriesResampler::new(1.0) // Resample to 1.0 time unit intervals
            .with_method(ResamplingMethod::Mean);

        let (resampled_times, resampled_values) = resampler.resample(&times, &values)?;

        // Should create regular grid from 0.0 to 3.0 with step 1.0
        assert_eq!(resampled_times.len(), 4);
        assert_abs_diff_eq!(resampled_times[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(resampled_times[3], 3.0, epsilon = 1e-10);

        // Values should be aggregated by mean within windows
        assert_eq!(resampled_values.len(), 4);

        Ok(())
    }

    #[test]
    fn test_multivariate_alignment() -> Result<()> {
        let series1_times = Array1::from(vec![0.0, 1.0, 2.0]);
        let series1_values = Array1::from(vec![10.0, 20.0, 30.0]);

        let series2_times = Array1::from(vec![0.5, 1.5, 2.5]);
        let series2_values = Array1::from(vec![15.0, 25.0, 35.0]);

        let series_data = vec![
            (series1_times, series1_values),
            (series2_times, series2_values),
        ];

        let aligner = MultiVariateTimeSeriesAligner::new()
            .with_interpolation_method(InterpolationMethod::Linear)
            .with_target_frequency(0.5);

        let (aligned_times, aligned_data) = aligner.align_series(&series_data)?;

        // Should have regular time grid with 0.5 step
        assert!(aligned_times.len() > 3);
        assert_eq!(aligned_data.dim().1, 2); // Two series

        // Check first and last time points
        assert_abs_diff_eq!(aligned_times[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(aligned_times[aligned_times.len() - 1], 2.5, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_empty_input() -> Result<()> {
        let times = Array1::zeros(0);
        let values = Array1::zeros(0);
        let target_times = Array1::from(vec![1.0, 2.0]);

        let interpolator = TimeSeriesInterpolator::new();
        let result = interpolator.interpolate(&times, &values, &target_times)?;

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);

        Ok(())
    }

    #[test]
    fn test_resampling_methods() -> Result<()> {
        let times = Array1::from(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
        let values = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Test different resampling methods
        let methods = vec![
            ResamplingMethod::Mean,
            ResamplingMethod::Sum,
            ResamplingMethod::Min,
            ResamplingMethod::Max,
            ResamplingMethod::First,
            ResamplingMethod::Last,
            ResamplingMethod::Count,
        ];

        for method in methods {
            let resampler = TimeSeriesResampler::new(0.5).with_method(method);
            let (resampled_times, resampled_values) = resampler.resample(&times, &values)?;

            assert!(resampled_times.len() > 0);
            assert_eq!(resampled_times.len(), resampled_values.len());

            // All values should be finite (not NaN)
            assert!(resampled_values.iter().all(|v| v.is_finite()));
        }

        Ok(())
    }
}
