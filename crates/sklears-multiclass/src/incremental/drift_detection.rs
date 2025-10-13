//! Concept Drift Detection for Online Learning
//!
//! Provides algorithms to detect when the underlying data distribution
//! changes over time, requiring model adaptation or retraining.

use sklears_core::{error::Result as SklResult, types::FloatBounds};
use std::collections::VecDeque;

/// Configuration for drift detection algorithms
#[derive(Debug, Clone)]
pub struct DriftDetectionConfig {
    /// Window size for monitoring predictions
    pub window_size: usize,
    /// Threshold for detecting drift
    pub drift_threshold: f64,
    /// Minimum number of samples before drift detection
    pub min_samples: usize,
    /// Warning threshold (lower than drift threshold)
    pub warning_threshold: f64,
}

impl Default for DriftDetectionConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            drift_threshold: 0.05,
            min_samples: 100,
            warning_threshold: 0.02,
        }
    }
}

/// Drift detection status
#[derive(Debug, Clone, PartialEq)]
pub enum DriftStatus {
    /// No drift detected
    Stable,
    /// Warning level - potential drift
    Warning,
    /// Drift detected - model adaptation needed
    Drift,
}

/// Trait for concept drift detection algorithms
pub trait DriftDetection<T: FloatBounds> {
    /// Update the detector with a new prediction and true label
    fn update(&mut self, prediction: T, actual: usize) -> SklResult<DriftStatus>;

    /// Get the current drift status
    fn status(&self) -> DriftStatus;

    /// Reset the detector state
    fn reset(&mut self);

    /// Get the current error rate
    fn error_rate(&self) -> f64;
}

/// Page-Hinkley Test for drift detection
#[derive(Debug, Clone)]
pub struct PageHinkleyDetector<T: FloatBounds> {
    config: DriftDetectionConfig,
    cumulative_sum: f64,
    min_cumsum: f64,
    num_samples: usize,
    current_status: DriftStatus,
    delta: f64,  // Magnitude of change to detect
    lambda: f64, // Threshold parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FloatBounds> PageHinkleyDetector<T> {
    /// Create a new Page-Hinkley detector
    pub fn new(config: DriftDetectionConfig) -> Self {
        Self {
            config,
            cumulative_sum: 0.0,
            min_cumsum: 0.0,
            num_samples: 0,
            current_status: DriftStatus::Stable,
            delta: 0.005,
            lambda: 50.0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(DriftDetectionConfig::default())
    }

    /// Set the magnitude of change to detect
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Set the threshold parameter
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }
}

impl<T: FloatBounds> DriftDetection<T> for PageHinkleyDetector<T> {
    fn update(&mut self, prediction: T, actual: usize) -> SklResult<DriftStatus> {
        // Simple error calculation (1 if incorrect, 0 if correct)
        let error = if prediction.to_f64().unwrap().round() as usize == actual {
            0.0
        } else {
            1.0
        };

        self.num_samples += 1;

        // Update cumulative sum
        self.cumulative_sum += error - self.delta;

        // Update minimum cumulative sum
        if self.cumulative_sum < self.min_cumsum {
            self.min_cumsum = self.cumulative_sum;
        }

        // Check for drift
        let ph_value = self.cumulative_sum - self.min_cumsum;

        if self.num_samples >= self.config.min_samples {
            if ph_value > self.lambda {
                self.current_status = DriftStatus::Drift;
            } else if ph_value > self.lambda * 0.5 {
                self.current_status = DriftStatus::Warning;
            } else {
                self.current_status = DriftStatus::Stable;
            }
        }

        Ok(self.current_status.clone())
    }

    fn status(&self) -> DriftStatus {
        self.current_status.clone()
    }

    fn reset(&mut self) {
        self.cumulative_sum = 0.0;
        self.min_cumsum = 0.0;
        self.num_samples = 0;
        self.current_status = DriftStatus::Stable;
    }

    fn error_rate(&self) -> f64 {
        if self.num_samples == 0 {
            0.0
        } else {
            // Approximate error rate based on cumulative sum
            (self.cumulative_sum / self.num_samples as f64).max(0.0)
        }
    }
}

/// ADWIN (ADaptive WINdowing) drift detector
#[derive(Debug, Clone)]
pub struct AdwinDetector<T: FloatBounds> {
    config: DriftDetectionConfig,
    window: VecDeque<f64>,
    current_status: DriftStatus,
    total_sum: f64,
    delta: f64, // Confidence parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FloatBounds> AdwinDetector<T> {
    /// Create a new ADWIN detector
    pub fn new(config: DriftDetectionConfig) -> Self {
        Self {
            config,
            window: VecDeque::new(),
            current_status: DriftStatus::Stable,
            total_sum: 0.0,
            delta: 0.002,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(DriftDetectionConfig::default())
    }

    /// Set the confidence parameter
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Check if drift occurred using ADWIN algorithm
    fn check_drift(&mut self) -> bool {
        if self.window.len() < self.config.min_samples {
            return false;
        }

        let n = self.window.len();
        let mean = self.total_sum / n as f64;

        // Simplified ADWIN: check if recent half differs significantly from older half
        if n >= 4 {
            let mid = n / 2;
            let recent_sum: f64 = self.window.iter().skip(mid).sum();
            let recent_mean = recent_sum / (n - mid) as f64;

            let old_sum: f64 = self.window.iter().take(mid).sum();
            let old_mean = old_sum / mid as f64;

            let diff = (recent_mean - old_mean).abs();
            let threshold = 2.0 * (2.0 * self.delta.ln() / n as f64).sqrt();

            return diff > threshold;
        }

        false
    }
}

impl<T: FloatBounds> DriftDetection<T> for AdwinDetector<T> {
    fn update(&mut self, prediction: T, actual: usize) -> SklResult<DriftStatus> {
        // Simple error calculation
        let error = if prediction.to_f64().unwrap().round() as usize == actual {
            0.0
        } else {
            1.0
        };

        // Add to window
        self.window.push_back(error);
        self.total_sum += error;

        // Maintain window size
        if self.window.len() > self.config.window_size {
            if let Some(removed) = self.window.pop_front() {
                self.total_sum -= removed;
            }
        }

        // Check for drift
        if self.check_drift() {
            self.current_status = DriftStatus::Drift;
            // Keep only recent half of the window
            let mid = self.window.len() / 2;
            self.window.drain(0..mid);
            self.total_sum = self.window.iter().sum();
        } else {
            let error_rate = self.error_rate();
            if error_rate > self.config.drift_threshold {
                self.current_status = DriftStatus::Warning;
            } else {
                self.current_status = DriftStatus::Stable;
            }
        }

        Ok(self.current_status.clone())
    }

    fn status(&self) -> DriftStatus {
        self.current_status.clone()
    }

    fn reset(&mut self) {
        self.window.clear();
        self.total_sum = 0.0;
        self.current_status = DriftStatus::Stable;
    }

    fn error_rate(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.total_sum / self.window.len() as f64
        }
    }
}

/// Comprehensive drift detector that combines multiple methods
#[derive(Debug, Clone)]
pub struct DriftDetector<T: FloatBounds> {
    page_hinkley: PageHinkleyDetector<T>,
    adwin: AdwinDetector<T>,
    config: DriftDetectionConfig,
}

impl<T: FloatBounds> DriftDetector<T> {
    /// Create a new comprehensive drift detector
    pub fn new(config: DriftDetectionConfig) -> Self {
        Self {
            page_hinkley: PageHinkleyDetector::new(config.clone()),
            adwin: AdwinDetector::new(config.clone()),
            config,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(DriftDetectionConfig::default())
    }
}

impl<T: FloatBounds> DriftDetection<T> for DriftDetector<T> {
    fn update(&mut self, prediction: T, actual: usize) -> SklResult<DriftStatus> {
        let ph_status = self.page_hinkley.update(prediction, actual)?;
        let adwin_status = self.adwin.update(prediction, actual)?;

        // Combine results - drift if either detects drift
        let combined_status = match (ph_status, adwin_status) {
            (DriftStatus::Drift, _) | (_, DriftStatus::Drift) => DriftStatus::Drift,
            (DriftStatus::Warning, _) | (_, DriftStatus::Warning) => DriftStatus::Warning,
            _ => DriftStatus::Stable,
        };

        Ok(combined_status)
    }

    fn status(&self) -> DriftStatus {
        let ph_status = self.page_hinkley.status();
        let adwin_status = self.adwin.status();

        match (ph_status, adwin_status) {
            (DriftStatus::Drift, _) | (_, DriftStatus::Drift) => DriftStatus::Drift,
            (DriftStatus::Warning, _) | (_, DriftStatus::Warning) => DriftStatus::Warning,
            _ => DriftStatus::Stable,
        }
    }

    fn reset(&mut self) {
        self.page_hinkley.reset();
        self.adwin.reset();
    }

    fn error_rate(&self) -> f64 {
        // Return the average error rate from both detectors
        (self.page_hinkley.error_rate() + self.adwin.error_rate()) / 2.0
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_detector_creation() {
        let detector: DriftDetector<f64> = DriftDetector::default();
        assert_eq!(detector.status(), DriftStatus::Stable);
    }

    #[test]
    fn test_page_hinkley_detector() {
        let mut detector: PageHinkleyDetector<f64> = PageHinkleyDetector::default();

        // No drift with correct predictions
        for i in 0..50 {
            let status = detector.update(1.0, 1).unwrap();
            assert_ne!(status, DriftStatus::Drift);
        }
    }

    #[test]
    fn test_adwin_detector() {
        let mut detector: AdwinDetector<f64> = AdwinDetector::default();

        // No drift with correct predictions
        for i in 0..50 {
            let status = detector.update(0.0, 0).unwrap();
            assert_ne!(status, DriftStatus::Drift);
        }
    }

    #[test]
    fn test_drift_status_enum() {
        assert_eq!(DriftStatus::Stable, DriftStatus::Stable);
        assert_ne!(DriftStatus::Stable, DriftStatus::Warning);
        assert_ne!(DriftStatus::Warning, DriftStatus::Drift);
    }
}
