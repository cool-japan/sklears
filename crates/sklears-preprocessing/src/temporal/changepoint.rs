//! Change point detection for time series
//!
//! This module provides various methods for detecting structural changes
//! or change points in time series data.

use scirs2_core::ndarray::{s, Array1};
use sklears_core::{
    error::Result,
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for ChangePointDetector
#[derive(Debug, Clone)]
pub struct ChangePointDetectorConfig {
    /// Detection method
    pub method: ChangePointMethod,
    /// Minimum segment length between change points
    pub min_segment_length: usize,
    /// Threshold for change point detection
    pub threshold: Float,
    /// Whether to return binary indicators or change point scores
    pub binary_output: bool,
}

/// Change point detection methods
#[derive(Debug, Clone, Copy)]
pub enum ChangePointMethod {
    /// CUSUM (Cumulative Sum) method
    CUSUM,
    /// Variance change detection
    VarianceChange,
    /// Mean change detection
    MeanChange,
    /// Simple difference-based detection
    SimpleDifference,
}

impl Default for ChangePointDetectorConfig {
    fn default() -> Self {
        Self {
            method: ChangePointMethod::CUSUM,
            min_segment_length: 5,
            threshold: 2.0,
            binary_output: true,
        }
    }
}

/// ChangePointDetector for detecting structural changes in time series
#[derive(Debug, Clone)]
pub struct ChangePointDetector<S> {
    config: ChangePointDetectorConfig,
    _phantom: PhantomData<S>,
}

impl ChangePointDetector<Untrained> {
    /// Create a new ChangePointDetector
    pub fn new() -> Self {
        Self {
            config: ChangePointDetectorConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Set the detection method
    pub fn method(mut self, method: ChangePointMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set the minimum segment length
    pub fn min_segment_length(mut self, min_segment_length: usize) -> Self {
        self.config.min_segment_length = min_segment_length;
        self
    }

    /// Set the detection threshold
    pub fn threshold(mut self, threshold: Float) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set whether to use binary output
    pub fn binary_output(mut self, binary_output: bool) -> Self {
        self.config.binary_output = binary_output;
        self
    }
}

impl ChangePointDetector<Trained> {
    /// CUSUM-based change point detection
    fn detect_cusum(&self, data: &Array1<Float>) -> Array1<Float> {
        let n = data.len();
        let mut scores = Array1::<Float>::zeros(n);

        if n < 2 {
            return scores;
        }

        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);

        if std < 1e-10 {
            return scores;
        }

        let mut cusum_pos = 0.0;
        let mut cusum_neg = 0.0;

        for i in 0..n {
            let standardized = (data[i] - mean) / std;

            cusum_pos = (cusum_pos + standardized - 0.5).max(0.0);
            cusum_neg = (cusum_neg - standardized - 0.5).max(0.0);

            scores[i] = cusum_pos.max(cusum_neg);
        }

        if self.config.binary_output {
            scores.mapv(|x| if x > self.config.threshold { 1.0 } else { 0.0 })
        } else {
            scores
        }
    }

    /// Variance-based change point detection
    fn detect_variance_change(&self, data: &Array1<Float>) -> Array1<Float> {
        let n = data.len();
        let window = self.config.min_segment_length;
        let mut scores = Array1::<Float>::zeros(n);

        if n < 2 * window {
            return scores;
        }

        for i in window..(n - window) {
            let left_window = data.slice(s![(i - window)..i]);
            let right_window = data.slice(s![i..(i + window)]);

            let left_var = left_window.var(0.0);
            let right_var = right_window.var(0.0);

            let ratio = if right_var > 1e-10 {
                left_var / right_var
            } else if left_var > 1e-10 {
                Float::INFINITY
            } else {
                1.0
            };

            scores[i] = (ratio.ln()).abs();
        }

        if self.config.binary_output {
            scores.mapv(|x| if x > self.config.threshold { 1.0 } else { 0.0 })
        } else {
            scores
        }
    }

    /// Mean-based change point detection
    fn detect_mean_change(&self, data: &Array1<Float>) -> Array1<Float> {
        let n = data.len();
        let window = self.config.min_segment_length;
        let mut scores = Array1::<Float>::zeros(n);

        if n < 2 * window {
            return scores;
        }

        for i in window..(n - window) {
            let left_window = data.slice(s![(i - window)..i]);
            let right_window = data.slice(s![i..(i + window)]);

            let left_mean = left_window.mean().unwrap_or(0.0);
            let right_mean = right_window.mean().unwrap_or(0.0);
            let pooled_std = ((left_window.var(0.0) + right_window.var(0.0)) / 2.0).sqrt();

            scores[i] = if pooled_std > 1e-10 {
                (left_mean - right_mean).abs() / pooled_std
            } else {
                0.0
            };
        }

        if self.config.binary_output {
            scores.mapv(|x| if x > self.config.threshold { 1.0 } else { 0.0 })
        } else {
            scores
        }
    }

    /// Simple difference-based change point detection
    fn detect_simple_difference(&self, data: &Array1<Float>) -> Array1<Float> {
        let n = data.len();
        let mut scores = Array1::<Float>::zeros(n);

        if n < 2 {
            return scores;
        }

        // Calculate first differences
        for i in 1..n {
            scores[i] = (data[i] - data[i - 1]).abs();
        }

        let threshold = if self.config.binary_output {
            scores.mean().unwrap_or(0.0) + self.config.threshold * scores.std(0.0)
        } else {
            0.0
        };

        if self.config.binary_output {
            scores.mapv(|x| if x > threshold { 1.0 } else { 0.0 })
        } else {
            scores
        }
    }
}

impl Default for ChangePointDetector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array1<Float>, ()> for ChangePointDetector<Untrained> {
    type Fitted = ChangePointDetector<Trained>;

    fn fit(self, _x: &Array1<Float>, _y: &()) -> Result<Self::Fitted> {
        Ok(ChangePointDetector {
            config: self.config,
            _phantom: PhantomData,
        })
    }
}

impl Transform<Array1<Float>, Array1<Float>> for ChangePointDetector<Trained> {
    fn transform(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let result = match self.config.method {
            ChangePointMethod::CUSUM => self.detect_cusum(x),
            ChangePointMethod::VarianceChange => self.detect_variance_change(x),
            ChangePointMethod::MeanChange => self.detect_mean_change(x),
            ChangePointMethod::SimpleDifference => self.detect_simple_difference(x),
        };

        Ok(result)
    }
}
