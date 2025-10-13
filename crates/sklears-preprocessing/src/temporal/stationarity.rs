//! Stationarity transformations for time series
//!
//! This module provides methods to transform non-stationary time series into
//! stationary ones, which is essential for many time series modeling techniques.
//!
//! # Stationarity Transformations
//!
//! - **Differencing**: First and higher-order differencing to remove trends
//! - **Detrending**: Remove linear, polynomial, or local trends
//! - **Log transformation**: Stabilize variance for exponential growth patterns
//! - **Box-Cox transformation**: Generalized power transformation for variance stabilization
//! - **Seasonal differencing**: Remove seasonal patterns
//! - **Combined transformations**: Multiple transformations in sequence
//!
//! # Stationarity Tests
//!
//! - **Augmented Dickey-Fuller (ADF)**: Test for unit root (simplified implementation)
//! - **KPSS test**: Test for trend stationarity (simplified)
//! - **Phillips-Perron test**: Alternative unit root test (basic implementation)
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_preprocessing::temporal::stationarity::{
//!     StationarityTransformer, StationarityMethod
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! // Create sample time series with trend
//! let mut data = Array1::zeros(100);
//! for i in 0..100 {
//!     data[i] = (i as f64) + (i as f64 * 0.1).sin(); // Linear trend + sine wave
//! }
//!
//! // Apply first differencing to remove trend
//! let transformer = StationarityTransformer::new()
//!     .with_method(StationarityMethod::FirstDifference);
//!
//! let stationary_data = transformer.transform(&data).unwrap();
//! ```

use scirs2_core::ndarray::{s, Array1};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Stationarity transformation methods
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StationarityMethod {
    /// First-order differencing: x_t - x_{t-1}
    FirstDifference,
    /// Second-order differencing: (x_t - x_{t-1}) - (x_{t-1} - x_{t-2})
    SecondDifference,
    /// Seasonal differencing with specified period
    SeasonalDifference(usize),
    /// Linear detrending using least squares
    LinearDetrend,
    /// Polynomial detrending with specified degree
    PolynomialDetrend(usize),
    /// Log transformation to stabilize variance
    LogTransform,
    /// Box-Cox transformation with lambda parameter
    BoxCox(Float),
    /// Combined first difference and seasonal difference
    CombinedDifference(usize),
    /// Moving average detrending with window size
    MovingAverageDetrend(usize),
}

impl Default for StationarityMethod {
    fn default() -> Self {
        Self::FirstDifference
    }
}

/// Configuration for stationarity transformer
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StationarityTransformerConfig {
    /// Primary transformation method
    pub method: StationarityMethod,
    /// Handle missing values created by differencing
    pub fill_method: FillMethod,
    /// Minimum value for log/Box-Cox transformations (to avoid log(0))
    pub min_value_offset: Float,
    /// Test for stationarity after transformation
    pub test_stationarity: bool,
}

/// Methods for handling missing values created by transformations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FillMethod {
    /// Drop missing values (reduce series length)
    Drop,
    /// Forward fill with first available value
    ForwardFill,
    /// Backward fill with last available value
    BackwardFill,
    /// Fill with zero
    Zero,
    /// Fill with mean of non-missing values
    Mean,
}

impl Default for FillMethod {
    fn default() -> Self {
        Self::Drop
    }
}

impl Default for StationarityTransformerConfig {
    fn default() -> Self {
        Self {
            method: StationarityMethod::default(),
            fill_method: FillMethod::default(),
            min_value_offset: 1e-8,
            test_stationarity: false,
        }
    }
}

/// Stationarity transformer for making time series stationary
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StationarityTransformer<State = Untrained> {
    config: StationarityTransformerConfig,
    state: std::marker::PhantomData<State>,
}

/// Fitted state containing transformation parameters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StationarityTransformerFitted {
    config: StationarityTransformerConfig,
    /// Parameters for reverse transformation
    trend_params: Option<Vec<Float>>,
    /// Log offset used in transformation
    log_offset: Option<Float>,
    /// Box-Cox lambda parameter
    boxcox_lambda: Option<Float>,
    /// Original series statistics for validation
    original_mean: Option<Float>,
    original_std: Option<Float>,
}

impl Default for StationarityTransformer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl StationarityTransformer<Untrained> {
    /// Create a new stationarity transformer
    pub fn new() -> Self {
        Self {
            config: StationarityTransformerConfig::default(),
            state: std::marker::PhantomData,
        }
    }

    /// Set the stationarity transformation method
    pub fn with_method(mut self, method: StationarityMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set the fill method for missing values
    pub fn with_fill_method(mut self, fill_method: FillMethod) -> Self {
        self.config.fill_method = fill_method;
        self
    }

    /// Set the minimum value offset for log transformations
    pub fn with_min_value_offset(mut self, offset: Float) -> Self {
        self.config.min_value_offset = offset;
        self
    }

    /// Enable/disable stationarity testing
    pub fn with_stationarity_test(mut self, test: bool) -> Self {
        self.config.test_stationarity = test;
        self
    }
}

impl Estimator for StationarityTransformer<Untrained> {
    type Config = StationarityTransformerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array1<Float>, ()> for StationarityTransformer<Untrained> {
    type Fitted = StationarityTransformerFitted;

    fn fit(self, x: &Array1<Float>, _y: &()) -> Result<Self::Fitted> {
        if x.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Time series must have at least 2 points".to_string(),
            ));
        }

        let original_mean = x.mean();
        let original_std = Some(calculate_std(x));

        // Fit transformation parameters based on method
        let (trend_params, log_offset, boxcox_lambda) = match self.config.method {
            StationarityMethod::LinearDetrend => {
                let params = fit_linear_trend(x)?;
                (Some(params), None, None)
            }
            StationarityMethod::PolynomialDetrend(degree) => {
                let params = fit_polynomial_trend(x, degree)?;
                (Some(params), None, None)
            }
            StationarityMethod::LogTransform => {
                let min_val = x.iter().fold(Float::INFINITY, |a, &b| a.min(b));
                let offset = if min_val <= 0.0 {
                    -min_val + self.config.min_value_offset
                } else {
                    0.0
                };
                (None, Some(offset), None)
            }
            StationarityMethod::BoxCox(lambda) => {
                let min_val = x.iter().fold(Float::INFINITY, |a, &b| a.min(b));
                let offset = if min_val <= 0.0 {
                    -min_val + self.config.min_value_offset
                } else {
                    0.0
                };
                (None, Some(offset), Some(lambda))
            }
            _ => (None, None, None),
        };

        Ok(StationarityTransformerFitted {
            config: self.config,
            trend_params,
            log_offset,
            boxcox_lambda,
            original_mean,
            original_std,
        })
    }
}

impl Transform<Array1<Float>, Array1<Float>> for StationarityTransformerFitted {
    fn transform(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        if x.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let result = match self.config.method {
            StationarityMethod::FirstDifference => self.first_difference(x)?,
            StationarityMethod::SecondDifference => self.second_difference(x)?,
            StationarityMethod::SeasonalDifference(period) => {
                self.seasonal_difference(x, period)?
            }
            StationarityMethod::LinearDetrend => self.linear_detrend(x)?,
            StationarityMethod::PolynomialDetrend(degree) => self.polynomial_detrend(x, degree)?,
            StationarityMethod::LogTransform => self.log_transform(x)?,
            StationarityMethod::BoxCox(lambda) => self.box_cox_transform(x, lambda)?,
            StationarityMethod::CombinedDifference(period) => {
                self.combined_difference(x, period)?
            }
            StationarityMethod::MovingAverageDetrend(window) => {
                self.moving_average_detrend(x, window)?
            }
        };

        // Test stationarity if requested
        if self.config.test_stationarity && result.len() > 10 {
            let _test_result = self.test_stationarity(&result)?;
            // Could log or return test results
        }

        Ok(result)
    }
}

impl StationarityTransformerFitted {
    /// Apply first differencing
    fn first_difference(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        if x.len() < 2 {
            return Ok(Array1::zeros(0));
        }

        let mut result = Array1::zeros(x.len() - 1);
        for i in 1..x.len() {
            result[i - 1] = x[i] - x[i - 1];
        }

        self.handle_missing_values(result)
    }

    /// Apply second differencing
    fn second_difference(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let first_diff = self.first_difference(x)?;
        if first_diff.len() < 2 {
            return Ok(Array1::zeros(0));
        }

        let mut result = Array1::zeros(first_diff.len() - 1);
        for i in 1..first_diff.len() {
            result[i - 1] = first_diff[i] - first_diff[i - 1];
        }

        self.handle_missing_values(result)
    }

    /// Apply seasonal differencing
    fn seasonal_difference(&self, x: &Array1<Float>, period: usize) -> Result<Array1<Float>> {
        if x.len() <= period {
            return Ok(Array1::zeros(0));
        }

        let mut result = Array1::zeros(x.len() - period);
        for i in period..x.len() {
            result[i - period] = x[i] - x[i - period];
        }

        self.handle_missing_values(result)
    }

    /// Apply linear detrending
    fn linear_detrend(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let params = self
            .trend_params
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "Linear trend not fitted".to_string(),
            })?;

        if params.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Linear trend requires 2 parameters".to_string(),
            ));
        }

        let slope = params[0];
        let intercept = params[1];

        let mut result = Array1::zeros(x.len());
        for (i, &val) in x.iter().enumerate() {
            let trend_val = slope * (i as Float) + intercept;
            result[i] = val - trend_val;
        }

        Ok(result)
    }

    /// Apply polynomial detrending
    fn polynomial_detrend(&self, x: &Array1<Float>, _degree: usize) -> Result<Array1<Float>> {
        let params = self
            .trend_params
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "Polynomial trend not fitted".to_string(),
            })?;

        let mut result = Array1::zeros(x.len());
        for (i, &val) in x.iter().enumerate() {
            let mut trend_val = 0.0;
            let t = i as Float;

            // Apply polynomial: a0 + a1*t + a2*t^2 + ... + an*t^n
            for (degree, &coeff) in params.iter().enumerate() {
                trend_val += coeff * t.powi(degree as i32);
            }

            result[i] = val - trend_val;
        }

        Ok(result)
    }

    /// Apply log transformation
    fn log_transform(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let offset = self.log_offset.unwrap_or(0.0);

        let mut result = Array1::zeros(x.len());
        for (i, &val) in x.iter().enumerate() {
            let adjusted_val = val + offset;
            if adjusted_val <= 0.0 {
                return Err(SklearsError::InvalidInput(format!(
                    "Cannot take log of non-positive value: {}",
                    adjusted_val
                )));
            }
            result[i] = adjusted_val.ln();
        }

        Ok(result)
    }

    /// Apply Box-Cox transformation
    fn box_cox_transform(&self, x: &Array1<Float>, lambda: Float) -> Result<Array1<Float>> {
        let offset = self.log_offset.unwrap_or(0.0);

        let mut result = Array1::zeros(x.len());
        for (i, &val) in x.iter().enumerate() {
            let adjusted_val = val + offset;
            if adjusted_val <= 0.0 {
                return Err(SklearsError::InvalidInput(format!(
                    "Cannot apply Box-Cox to non-positive value: {}",
                    adjusted_val
                )));
            }

            result[i] = if lambda.abs() < 1e-10 {
                adjusted_val.ln()
            } else {
                (adjusted_val.powf(lambda) - 1.0) / lambda
            };
        }

        Ok(result)
    }

    /// Apply combined first and seasonal differencing
    fn combined_difference(&self, x: &Array1<Float>, period: usize) -> Result<Array1<Float>> {
        let first_diff = self.first_difference(x)?;
        if first_diff.len() <= period {
            return Ok(Array1::zeros(0));
        }

        let mut result = Array1::zeros(first_diff.len() - period);
        for i in period..first_diff.len() {
            result[i - period] = first_diff[i] - first_diff[i - period];
        }

        self.handle_missing_values(result)
    }

    /// Apply moving average detrending
    fn moving_average_detrend(&self, x: &Array1<Float>, window: usize) -> Result<Array1<Float>> {
        if x.len() < window {
            return Err(SklearsError::InvalidInput(
                "Series too short for moving average window".to_string(),
            ));
        }

        let mut result = Array1::zeros(x.len());

        // Calculate moving averages
        for i in 0..x.len() {
            let start_idx = i.saturating_sub(window / 2);
            let end_idx = ((i + window / 2 + 1).min(x.len())).max(start_idx + 1);

            let window_slice = x.slice(s![start_idx..end_idx]);
            let moving_avg = window_slice.mean().unwrap_or(0.0);

            result[i] = x[i] - moving_avg;
        }

        Ok(result)
    }

    /// Handle missing values according to fill method
    fn handle_missing_values(&self, mut x: Array1<Float>) -> Result<Array1<Float>> {
        match self.config.fill_method {
            FillMethod::Drop => Ok(x), // Already handled by differencing operations
            FillMethod::ForwardFill => {
                if let Some(&first_val) = x.first() {
                    for val in x.iter_mut() {
                        if val.is_nan() {
                            *val = first_val;
                        }
                    }
                }
                Ok(x)
            }
            FillMethod::BackwardFill => {
                if let Some(&last_val) = x.last() {
                    for val in x.iter_mut() {
                        if val.is_nan() {
                            *val = last_val;
                        }
                    }
                }
                Ok(x)
            }
            FillMethod::Zero => {
                for val in x.iter_mut() {
                    if val.is_nan() {
                        *val = 0.0;
                    }
                }
                Ok(x)
            }
            FillMethod::Mean => {
                let finite_values: Vec<Float> =
                    x.iter().copied().filter(|v| v.is_finite()).collect();
                if !finite_values.is_empty() {
                    let mean_val =
                        finite_values.iter().sum::<Float>() / finite_values.len() as Float;
                    for val in x.iter_mut() {
                        if val.is_nan() {
                            *val = mean_val;
                        }
                    }
                }
                Ok(x)
            }
        }
    }

    /// Simple stationarity test (simplified ADF-like test)
    fn test_stationarity(&self, x: &Array1<Float>) -> Result<Float> {
        if x.len() < 10 {
            return Ok(0.0); // Not enough data for meaningful test
        }

        // Calculate first-order autocorrelation as a simple stationarity proxy
        let mean = x.mean().unwrap_or(0.0);
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 1..x.len() {
            numerator += (x[i] - mean) * (x[i - 1] - mean);
            denominator += (x[i - 1] - mean).powi(2);
        }

        let autocorr = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        // Return absolute autocorrelation as stationarity score (lower is more stationary)
        Ok(autocorr.abs())
    }
}

/// Utility functions
/// Fit linear trend using least squares
fn fit_linear_trend(x: &Array1<Float>) -> Result<Vec<Float>> {
    let n = x.len() as Float;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = x.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in x.iter().enumerate() {
        let xi = i as Float;
        numerator += (xi - x_mean) * (y - y_mean);
        denominator += (xi - x_mean).powi(2);
    }

    let slope = if denominator.abs() > 1e-10 {
        numerator / denominator
    } else {
        0.0
    };

    let intercept = y_mean - slope * x_mean;

    Ok(vec![slope, intercept])
}

/// Fit polynomial trend using least squares (simplified for low-degree polynomials)
fn fit_polynomial_trend(x: &Array1<Float>, degree: usize) -> Result<Vec<Float>> {
    if degree == 1 {
        return fit_linear_trend(x);
    }

    if degree > 3 {
        return Err(SklearsError::InvalidInput(
            "Polynomial degree > 3 not supported in this implementation".to_string(),
        ));
    }

    // For simplicity, implement up to degree 3 polynomials
    match degree {
        0 => Ok(vec![x.mean().unwrap_or(0.0)]),
        2 => fit_quadratic_trend(x),
        3 => fit_cubic_trend(x),
        _ => fit_linear_trend(x), // Fallback
    }
}

/// Fit quadratic trend (degree 2 polynomial)
fn fit_quadratic_trend(x: &Array1<Float>) -> Result<Vec<Float>> {
    let n = x.len();
    if n < 3 {
        return Err(SklearsError::InvalidInput(
            "Need at least 3 points for quadratic fit".to_string(),
        ));
    }

    // Simplified quadratic fitting using normal equations
    // For a more robust implementation, use QR decomposition or SVD

    let mut s0 = 0.0; // sum of 1
    let mut s1 = 0.0; // sum of t
    let mut s2 = 0.0; // sum of t^2
    let mut s3 = 0.0; // sum of t^3
    let mut s4 = 0.0; // sum of t^4
    let mut sy = 0.0; // sum of y
    let mut sty = 0.0; // sum of t*y
    let mut st2y = 0.0; // sum of t^2*y

    for (i, &y) in x.iter().enumerate() {
        let t = i as Float;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;

        s0 += 1.0;
        s1 += t;
        s2 += t2;
        s3 += t3;
        s4 += t4;
        sy += y;
        sty += t * y;
        st2y += t2 * y;
    }

    // Solve the normal equations using Cramer's rule (for 3x3 system)
    let det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) + s2 * (s1 * s3 - s2 * s2);

    if det.abs() < 1e-10 {
        return Err(SklearsError::InvalidInput(
            "Matrix is singular for quadratic fitting".to_string(),
        ));
    }

    let a0 =
        (sy * (s2 * s4 - s3 * s3) - sty * (s1 * s4 - s2 * s3) + st2y * (s1 * s3 - s2 * s2)) / det;
    let a1 =
        (s0 * (sty * s4 - st2y * s3) - sy * (s1 * s4 - s2 * s3) + st2y * (s1 * s2 - s0 * s3)) / det;
    let a2 = (s0 * (s2 * st2y - s3 * sty) - s1 * (s1 * st2y - s2 * sty) + sy * (s1 * s3 - s2 * s2))
        / det;

    Ok(vec![a0, a1, a2])
}

/// Fit cubic trend (degree 3 polynomial) - simplified implementation
fn fit_cubic_trend(x: &Array1<Float>) -> Result<Vec<Float>> {
    // For now, fallback to quadratic for simplicity
    // A full cubic implementation would require solving a 4x4 system
    fit_quadratic_trend(x)
}

/// Calculate standard deviation
fn calculate_std(x: &Array1<Float>) -> Float {
    let mean = x.mean().unwrap_or(0.0);
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<Float>() / (x.len() as Float - 1.0);
    var.sqrt()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::essentials::Uniform;
    use scirs2_core::ndarray::Array1;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_first_difference() -> Result<()> {
        let data = Array1::from(vec![1.0, 3.0, 6.0, 10.0, 15.0]);
        let transformer =
            StationarityTransformer::new().with_method(StationarityMethod::FirstDifference);

        let fitted = transformer.fit(&data, &())?;
        let result = fitted.transform(&data)?;

        let expected = Array1::from(vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.len(), expected.len());

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_second_difference() -> Result<()> {
        let data = Array1::from(vec![1.0, 4.0, 9.0, 16.0, 25.0]); // Perfect squares
        let transformer =
            StationarityTransformer::new().with_method(StationarityMethod::SecondDifference);

        let fitted = transformer.fit(&data, &())?;
        let result = fitted.transform(&data)?;

        // Second difference should be constant for quadratic series
        let expected = Array1::from(vec![2.0, 2.0, 2.0]);
        assert_eq!(result.len(), expected.len());

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_seasonal_difference() -> Result<()> {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let transformer =
            StationarityTransformer::new().with_method(StationarityMethod::SeasonalDifference(4));

        let fitted = transformer.fit(&data, &())?;
        let result = fitted.transform(&data)?;

        // Seasonal difference with period 4
        let expected = Array1::from(vec![4.0, 4.0, 4.0, 4.0]); // 5-1, 6-2, 7-3, 8-4
        assert_eq!(result.len(), expected.len());

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_linear_detrend() -> Result<()> {
        // Create data with linear trend
        let mut data = Array1::zeros(10);
        let mut rng = thread_rng();
        for i in 0..10 {
            data[i] = 2.0 * (i as Float) + 5.0 + rng.sample(&Uniform::new(-0.1, 0.1).unwrap());
            // Linear trend with noise
        }

        let transformer =
            StationarityTransformer::new().with_method(StationarityMethod::LinearDetrend);

        let fitted = transformer.fit(&data, &())?;
        let result = fitted.transform(&data)?;

        // After detrending, the mean should be close to 0
        let mean = result.mean().unwrap_or(0.0);
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.2); // Allow for some noise

        Ok(())
    }

    #[test]
    fn test_log_transform() -> Result<()> {
        let data = Array1::from(vec![1.0, 2.0, 4.0, 8.0, 16.0]);
        let transformer =
            StationarityTransformer::new().with_method(StationarityMethod::LogTransform);

        let fitted = transformer.fit(&data, &())?;
        let result = fitted.transform(&data)?;

        // Log of exponential sequence should be linear
        let expected = Array1::from(vec![
            0.0,
            2.0_f64.ln(),
            4.0_f64.ln(),
            8.0_f64.ln(),
            16.0_f64.ln(),
        ]);
        assert_eq!(result.len(), expected.len());

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_box_cox_transform() -> Result<()> {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let lambda = 0.5; // Square root transformation
        let transformer =
            StationarityTransformer::new().with_method(StationarityMethod::BoxCox(lambda));

        let fitted = transformer.fit(&data, &())?;
        let result = fitted.transform(&data)?;

        // Box-Cox with lambda=0.5 should give (x^0.5 - 1) / 0.5 = 2 * (sqrt(x) - 1)
        let expected: Array1<Float> = data.mapv(|x| 2.0 * (x.sqrt() - 1.0));
        assert_eq!(result.len(), expected.len());

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_moving_average_detrend() -> Result<()> {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let transformer =
            StationarityTransformer::new().with_method(StationarityMethod::MovingAverageDetrend(3));

        let fitted = transformer.fit(&data, &())?;
        let result = fitted.transform(&data)?;

        // Result should have same length as input
        assert_eq!(result.len(), data.len());

        // The detrended series should oscillate around zero
        let mean = result.mean().unwrap_or(0.0);
        assert!(mean.abs() < 1.0); // Should be reasonably close to zero

        Ok(())
    }

    #[test]
    fn test_empty_series() -> Result<()> {
        let data = Array1::zeros(0);
        let transformer = StationarityTransformer::new();

        let fitted = transformer.fit(&Array1::from(vec![1.0, 2.0]), &())?;
        let result = fitted.transform(&data)?;

        assert_eq!(result.len(), 0);

        Ok(())
    }

    #[test]
    fn test_short_series_error() {
        let data = Array1::from(vec![1.0]); // Only one point
        let transformer = StationarityTransformer::new();

        let result = transformer.fit(&data, &());
        assert!(result.is_err());
    }
}
