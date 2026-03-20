//! Trend detection and analysis for time series
//!
//! This module provides various methods for detecting and quantifying trends
//! in time series data.

use scirs2_core::ndarray::{s, Array1};
use sklears_core::{
    error::Result,
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for TrendDetector
#[derive(Debug, Clone)]
pub struct TrendDetectorConfig {
    /// Trend detection method
    pub method: TrendMethod,
    /// Window size for local trend calculation
    pub window_size: usize,
    /// Polynomial degree for polynomial trend fitting
    pub polynomial_degree: usize,
}

/// Trend detection methods
#[derive(Debug, Clone, Copy)]
pub enum TrendMethod {
    /// Linear trend using least squares
    Linear,
    /// Polynomial trend fitting
    Polynomial,
    /// Local linear trends in sliding windows
    LocalLinear,
    /// Mann-Kendall trend test
    MannKendall,
}

impl Default for TrendDetectorConfig {
    fn default() -> Self {
        Self {
            method: TrendMethod::Linear,
            window_size: 10,
            polynomial_degree: 2,
        }
    }
}

/// TrendDetector for detecting and quantifying trends in time series
#[derive(Debug, Clone)]
pub struct TrendDetector<S> {
    config: TrendDetectorConfig,
    _phantom: PhantomData<S>,
}

impl TrendDetector<Untrained> {
    /// Create a new TrendDetector
    pub fn new() -> Self {
        Self {
            config: TrendDetectorConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Set the trend detection method
    pub fn method(mut self, method: TrendMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set the window size for local trend calculation
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = window_size;
        self
    }

    /// Set the polynomial degree
    pub fn polynomial_degree(mut self, polynomial_degree: usize) -> Self {
        self.config.polynomial_degree = polynomial_degree;
        self
    }
}

impl TrendDetector<Trained> {
    /// Calculate linear trend slope using least squares
    fn calculate_linear_trend(&self, data: &Array1<Float>) -> Float {
        let n = data.len() as Float;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = data.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as Float;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Calculate Mann-Kendall trend statistic
    fn calculate_mann_kendall(&self, data: &Array1<Float>) -> Float {
        let n = data.len();
        let mut s = 0i32;

        for i in 0..n {
            for j in (i + 1)..n {
                if data[j] > data[i] {
                    s += 1;
                } else if data[j] < data[i] {
                    s -= 1;
                }
            }
        }

        // Normalize by maximum possible value
        let max_s = (n * (n - 1) / 2) as i32;
        if max_s > 0 {
            s as Float / max_s as Float
        } else {
            0.0
        }
    }

    /// Calculate local linear trends in sliding windows
    fn calculate_local_trends(&self, data: &Array1<Float>) -> Array1<Float> {
        let n = data.len();
        let window_size = self.config.window_size.min(n);
        let mut trends = Array1::<Float>::zeros(n);

        for i in 0..n {
            let start = if i >= window_size / 2 {
                (i - window_size / 2).max(0)
            } else {
                0
            };
            let end = (start + window_size).min(n);
            let window = data.slice(s![start..end]);
            trends[i] = self.calculate_linear_trend(&window.to_owned());
        }

        trends
    }
}

impl Default for TrendDetector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array1<Float>, ()> for TrendDetector<Untrained> {
    type Fitted = TrendDetector<Trained>;

    fn fit(self, _x: &Array1<Float>, _y: &()) -> Result<Self::Fitted> {
        Ok(TrendDetector {
            config: self.config,
            _phantom: PhantomData,
        })
    }
}

impl Transform<Array1<Float>, Array1<Float>> for TrendDetector<Trained> {
    fn transform(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        match self.config.method {
            TrendMethod::Linear => {
                let slope = self.calculate_linear_trend(x);
                Ok(Array1::from_elem(x.len(), slope))
            }
            TrendMethod::MannKendall => {
                let mk_stat = self.calculate_mann_kendall(x);
                Ok(Array1::from_elem(x.len(), mk_stat))
            }
            TrendMethod::LocalLinear => Ok(self.calculate_local_trends(x)),
            TrendMethod::Polynomial => {
                // Simplified polynomial trend - just return linear for now
                let slope = self.calculate_linear_trend(x);
                Ok(Array1::from_elem(x.len(), slope))
            }
        }
    }
}
