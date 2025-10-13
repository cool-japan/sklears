//! Streaming infrastructure for incremental decision trees
//!
//! This module provides the core streaming infrastructure for incremental decision tree
//! algorithms, including data buffering, concept drift detection, and adaptive windowing.
//! These components are essential for handling streaming data scenarios where data
//! distribution may change over time.
//!
//! ## Core Components
//!
//! - **IncrementalTreeConfig**: Configuration for incremental tree building
//! - **StreamingBuffer**: Efficient buffering for streaming data with time-window operations
//! - **ConceptDriftDetector**: Basic concept drift detection using accuracy tracking
//! - **AdwinDetector**: ADWIN (Adaptive Windowing) for automatic window size adjustment
//! - **AdaptiveConceptDriftDetector**: Enhanced drift detection combining multiple approaches

use super::simd_operations as simd_tree;
use crate::{DecisionTreeConfig, Untrained};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;

/// Configuration for incremental tree building
#[derive(Debug, Clone)]
pub struct IncrementalTreeConfig {
    /// Base decision tree configuration
    pub base_config: DecisionTreeConfig,
    /// Window size for concept drift detection
    pub window_size: usize,
    /// Minimum number of samples before starting tree construction
    pub min_samples_before_build: usize,
    /// Concept drift detection threshold
    pub drift_threshold: f64,
    /// Enable concept drift detection
    pub enable_drift_detection: bool,
    /// Buffer size for incoming samples
    pub buffer_size: usize,
    /// Update frequency (rebuild tree every N samples)
    pub update_frequency: usize,
}

impl Default for IncrementalTreeConfig {
    fn default() -> Self {
        Self {
            base_config: DecisionTreeConfig::default(),
            window_size: 1000,
            min_samples_before_build: 100,
            drift_threshold: 0.05,
            enable_drift_detection: true,
            buffer_size: 10000,
            update_frequency: 100,
        }
    }
}

/// Streaming data buffer for incremental learning
#[derive(Debug, Clone)]
pub struct StreamingBuffer {
    /// Feature data buffer
    pub x_buffer: VecDeque<Vec<f64>>,
    /// Target data buffer
    pub y_buffer: VecDeque<f64>,
    /// Sample weights buffer
    pub weight_buffer: VecDeque<f64>,
    /// Timestamps for concept drift detection
    pub timestamps: VecDeque<u64>,
    /// Maximum buffer size
    pub max_size: usize,
}

impl StreamingBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            x_buffer: VecDeque::with_capacity(max_size),
            y_buffer: VecDeque::with_capacity(max_size),
            weight_buffer: VecDeque::with_capacity(max_size),
            timestamps: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a new sample to the buffer
    pub fn add_sample(&mut self, x: Vec<f64>, y: f64, weight: f64, timestamp: u64) {
        // Remove oldest samples if buffer is full
        if self.x_buffer.len() >= self.max_size {
            self.x_buffer.pop_front();
            self.y_buffer.pop_front();
            self.weight_buffer.pop_front();
            self.timestamps.pop_front();
        }

        self.x_buffer.push_back(x);
        self.y_buffer.push_back(y);
        self.weight_buffer.push_back(weight);
        self.timestamps.push_back(timestamp);
    }

    /// Get the current buffer size
    pub fn len(&self) -> usize {
        self.x_buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.x_buffer.is_empty()
    }

    /// Convert buffer to arrays for training
    pub fn to_arrays(&self) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        if self.is_empty() {
            return Err(SklearsError::InvalidInput("Buffer is empty".to_string()));
        }

        let n_samples = self.len();
        let n_features = self.x_buffer[0].len();

        // Convert feature data
        let mut x_data = Array2::zeros((n_samples, n_features));
        for (i, sample) in self.x_buffer.iter().enumerate() {
            for (j, &value) in sample.iter().enumerate() {
                x_data[[i, j]] = value;
            }
        }

        // Convert target data
        let y_data = Array1::from_vec(self.y_buffer.iter().cloned().collect());

        // Convert weights
        let weights = Array1::from_vec(self.weight_buffer.iter().cloned().collect());

        Ok((x_data, y_data, weights))
    }

    /// Get recent samples within a time window
    pub fn get_recent_samples(&self, window_duration: u64) -> Result<(Array2<f64>, Array1<f64>)> {
        if self.is_empty() {
            return Err(SklearsError::InvalidInput("Buffer is empty".to_string()));
        }

        let latest_timestamp = *self.timestamps.back().unwrap();
        let cutoff_timestamp = latest_timestamp.saturating_sub(window_duration);

        // Find samples within the window
        let mut recent_x = Vec::new();
        let mut recent_y = Vec::new();

        for (i, &timestamp) in self.timestamps.iter().enumerate() {
            if timestamp >= cutoff_timestamp {
                recent_x.push(self.x_buffer[i].clone());
                recent_y.push(self.y_buffer[i]);
            }
        }

        if recent_x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No recent samples found".to_string(),
            ));
        }

        let n_samples = recent_x.len();
        let n_features = recent_x[0].len();

        let mut x_array = Array2::zeros((n_samples, n_features));
        for (i, sample) in recent_x.iter().enumerate() {
            for (j, &value) in sample.iter().enumerate() {
                x_array[[i, j]] = value;
            }
        }

        let y_array = Array1::from_vec(recent_y);

        Ok((x_array, y_array))
    }
}

/// Concept drift detector for streaming data
#[derive(Debug, Clone)]
pub struct ConceptDriftDetector {
    /// Recent accuracy values for drift detection
    pub accuracy_history: VecDeque<f64>,
    /// Reference accuracy (baseline)
    pub reference_accuracy: f64,
    /// Window size for drift detection
    pub window_size: usize,
    /// Drift detection threshold
    pub threshold: f64,
    /// Number of consecutive drops needed to signal drift
    pub consecutive_drops: usize,
    /// Current number of consecutive accuracy drops
    pub current_drops: usize,
}

impl ConceptDriftDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            accuracy_history: VecDeque::with_capacity(window_size),
            reference_accuracy: 0.0,
            window_size,
            threshold,
            consecutive_drops: 3,
            current_drops: 0,
        }
    }

    /// Update the detector with new accuracy value
    pub fn update(&mut self, accuracy: f64) -> bool {
        self.accuracy_history.push_back(accuracy);

        if self.accuracy_history.len() > self.window_size {
            self.accuracy_history.pop_front();
        }

        // Initialize reference accuracy
        if self.reference_accuracy == 0.0 && self.accuracy_history.len() >= self.window_size / 2 {
            self.reference_accuracy =
                self.accuracy_history.iter().sum::<f64>() / self.accuracy_history.len() as f64;
        }

        // Check for drift
        if self.reference_accuracy > 0.0 {
            let recent_accuracy = self
                .accuracy_history
                .iter()
                .rev()
                .take(self.window_size / 4)
                .sum::<f64>()
                / (self.window_size / 4) as f64;

            if self.reference_accuracy - recent_accuracy > self.threshold {
                self.current_drops += 1;
                if self.current_drops >= self.consecutive_drops {
                    self.current_drops = 0;
                    self.reference_accuracy = recent_accuracy; // Update reference
                    return true; // Drift detected
                }
            } else {
                self.current_drops = 0;
            }
        }

        false
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.accuracy_history.clear();
        self.reference_accuracy = 0.0;
        self.current_drops = 0;
    }

    /// Check if drift was detected (for compatibility with adaptive detector)
    pub fn is_drift_detected(&self) -> bool {
        self.current_drops >= self.consecutive_drops
    }

    /// Get current window statistics
    pub fn get_window_statistics(&self) -> (usize, f64, usize) {
        (
            self.window_size,
            self.reference_accuracy,
            self.current_drops,
        )
    }
}

/// ADWIN (ADaptive WINdowing) for automatic window size adjustment
///
/// ADWIN automatically adjusts the window size to keep only recent data
/// that appears to be from the same distribution. It uses statistical tests
/// to detect when the data distribution changes and shrinks the window accordingly.
#[derive(Debug, Clone)]
pub struct AdwinDetector {
    /// Window of recent values
    window: VecDeque<f64>,
    /// Confidence level for statistical tests (default: 0.95)
    confidence: f64,
    /// Minimum window size to maintain
    min_window_size: usize,
    /// Maximum window size to prevent memory overflow
    max_window_size: usize,
    /// Sum of values in the window
    window_sum: f64,
    /// Sum of squared values for variance calculation
    window_sum_squares: f64,
    /// Flag indicating if concept drift was detected
    drift_detected: bool,
    /// Number of cuts (window adjustments) performed
    cuts_performed: usize,
}

impl Default for AdwinDetector {
    fn default() -> Self {
        Self::new(0.95, 10, 10000)
    }
}

impl AdwinDetector {
    /// Create a new ADWIN detector
    pub fn new(confidence: f64, min_window_size: usize, max_window_size: usize) -> Self {
        Self {
            window: VecDeque::new(),
            confidence,
            min_window_size,
            max_window_size,
            window_sum: 0.0,
            window_sum_squares: 0.0,
            drift_detected: false,
            cuts_performed: 0,
        }
    }

    /// Add a new value and check for concept drift
    pub fn update(&mut self, value: f64) -> bool {
        self.drift_detected = false;

        // Add new value to window
        self.add_to_window(value);

        // Check if window is large enough for statistical tests
        if self.window.len() < self.min_window_size {
            return false;
        }

        // Perform ADWIN algorithm
        self.perform_adwin_test()
    }

    /// Add a value to the window, managing size constraints
    fn add_to_window(&mut self, value: f64) {
        // Remove oldest values if at maximum capacity
        while self.window.len() >= self.max_window_size {
            if let Some(old_value) = self.window.pop_front() {
                self.window_sum -= old_value;
                self.window_sum_squares -= old_value * old_value;
            }
        }

        // Add new value
        self.window.push_back(value);
        self.window_sum += value;
        self.window_sum_squares += value * value;
    }

    /// Perform the ADWIN statistical test for concept drift
    fn perform_adwin_test(&mut self) -> bool {
        let n = self.window.len();
        if n < self.min_window_size {
            return false;
        }

        // Try different split points in the window
        for cut_point in self.min_window_size..=(n - self.min_window_size) {
            if self.test_cut_point(cut_point) {
                // Cut detected - remove old part of window
                self.perform_cut(cut_point);
                self.drift_detected = true;
                self.cuts_performed += 1;
                return true;
            }
        }

        false
    }

    /// Test if a cut point indicates significant change
    fn test_cut_point(&self, cut_point: usize) -> bool {
        let n = self.window.len();
        let n0 = cut_point;
        let n1 = n - cut_point;

        if n0 < self.min_window_size || n1 < self.min_window_size {
            return false;
        }

        // Calculate means and variances for both parts
        let (mean0, var0) = self.calculate_stats_for_range(0, cut_point);
        let (mean1, var1) = self.calculate_stats_for_range(cut_point, n);

        // Calculate the bound for the statistical test
        let bound = self.calculate_bound(n0, n1, var0, var1);

        // Test if the difference in means is significant
        (mean0 - mean1).abs() > bound
    }

    /// Calculate mean and variance for a range in the window using SIMD acceleration
    fn calculate_stats_for_range(&self, start: usize, end: usize) -> (f64, f64) {
        let window_vec: Vec<f64> = self.window.iter().cloned().collect();
        simd_tree::simd_calculate_range_stats(&window_vec, start, end)
    }

    /// Calculate the statistical bound for the ADWIN test using SIMD acceleration
    fn calculate_bound(&self, n0: usize, n1: usize, var0: f64, var1: f64) -> f64 {
        let alpha = 1.0 - self.confidence;
        simd_tree::simd_adwin_bound_calculation(n0, n1, var0, var1, alpha)
    }

    /// Perform the cut by removing the old part of the window
    fn perform_cut(&mut self, cut_point: usize) {
        // Remove elements before the cut point
        for _ in 0..cut_point {
            if let Some(old_value) = self.window.pop_front() {
                self.window_sum -= old_value;
                self.window_sum_squares -= old_value * old_value;
            }
        }
    }

    /// Check if drift was detected in the last update
    pub fn drift_detected(&self) -> bool {
        self.drift_detected
    }

    /// Get current window size
    pub fn window_size(&self) -> usize {
        self.window.len()
    }

    /// Get number of cuts performed
    pub fn cuts_performed(&self) -> usize {
        self.cuts_performed
    }

    /// Get current window mean
    pub fn window_mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.window_sum / self.window.len() as f64
        }
    }

    /// Get current window variance
    pub fn window_variance(&self) -> f64 {
        let n = self.window.len();
        if n <= 1 {
            return 0.0;
        }

        let mean = self.window_mean();
        let variance = (self.window_sum_squares - self.window_sum * mean) / (n - 1) as f64;
        variance.max(0.0)
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.window.clear();
        self.window_sum = 0.0;
        self.window_sum_squares = 0.0;
        self.drift_detected = false;
        self.cuts_performed = 0;
    }

    /// Get the window contents (for debugging/analysis)
    pub fn get_window(&self) -> Vec<f64> {
        self.window.iter().cloned().collect()
    }

    /// Set new confidence level
    pub fn set_confidence(&mut self, confidence: f64) {
        self.confidence = confidence.clamp(0.5, 0.999);
    }

    /// Check if the window is ready for drift detection
    pub fn is_ready(&self) -> bool {
        self.window.len() >= self.min_window_size
    }

    /// Get adaptive window statistics
    pub fn get_statistics(&self) -> AdwinStatistics {
        AdwinStatistics {
            window_size: self.window_size(),
            mean: self.window_mean(),
            variance: self.window_variance(),
            cuts_performed: self.cuts_performed,
            confidence: self.confidence,
            min_window_size: self.min_window_size,
            max_window_size: self.max_window_size,
        }
    }
}

/// Statistics for ADWIN detector
#[derive(Debug, Clone)]
pub struct AdwinStatistics {
    /// Current window size
    pub window_size: usize,
    /// Current window mean
    pub mean: f64,
    /// Current window variance
    pub variance: f64,
    /// Number of cuts performed
    pub cuts_performed: usize,
    /// Confidence level
    pub confidence: f64,
    /// Minimum window size
    pub min_window_size: usize,
    /// Maximum window size
    pub max_window_size: usize,
}

/// Enhanced concept drift detector with adaptive windowing
#[derive(Debug, Clone)]
pub struct AdaptiveConceptDriftDetector {
    /// ADWIN detector for accuracy values
    adwin_detector: AdwinDetector,
    /// Traditional fixed-window detector for comparison
    fixed_detector: ConceptDriftDetector,
    /// Enable hybrid mode (use both detectors)
    hybrid_mode: bool,
    /// Minimum samples before starting detection
    min_samples: usize,
    /// Sample counter
    sample_count: usize,
}

impl AdaptiveConceptDriftDetector {
    pub fn new(
        confidence: f64,
        min_window_size: usize,
        max_window_size: usize,
        fixed_window_size: usize,
        fixed_threshold: f64,
        hybrid_mode: bool,
    ) -> Self {
        Self {
            adwin_detector: AdwinDetector::new(confidence, min_window_size, max_window_size),
            fixed_detector: ConceptDriftDetector::new(fixed_window_size, fixed_threshold),
            hybrid_mode,
            min_samples: min_window_size,
            sample_count: 0,
        }
    }

    /// Update with new accuracy value and check for drift
    pub fn update(&mut self, accuracy: f64) -> bool {
        self.sample_count += 1;

        if self.sample_count < self.min_samples {
            return false;
        }

        let adwin_drift = self.adwin_detector.update(accuracy);

        if self.hybrid_mode {
            let fixed_drift = self.fixed_detector.update(accuracy);
            // Return true if either detector finds drift
            adwin_drift || fixed_drift
        } else {
            adwin_drift
        }
    }

    /// Check if drift was detected
    pub fn drift_detected(&self) -> bool {
        if self.hybrid_mode {
            self.adwin_detector.drift_detected() || self.fixed_detector.is_drift_detected()
        } else {
            self.adwin_detector.drift_detected()
        }
    }

    /// Get current adaptive window size
    pub fn adaptive_window_size(&self) -> usize {
        self.adwin_detector.window_size()
    }

    /// Get current window mean
    pub fn window_mean(&self) -> f64 {
        self.adwin_detector.window_mean()
    }

    /// Get ADWIN statistics
    pub fn get_adwin_statistics(&self) -> AdwinStatistics {
        self.adwin_detector.get_statistics()
    }

    /// Reset both detectors
    pub fn reset(&mut self) {
        self.adwin_detector.reset();
        self.fixed_detector.reset();
        self.sample_count = 0;
    }

    /// Check if the detector is ready
    pub fn is_ready(&self) -> bool {
        self.sample_count >= self.min_samples && self.adwin_detector.is_ready()
    }
}

impl Default for AdaptiveConceptDriftDetector {
    fn default() -> Self {
        Self::new(
            0.95,  // confidence
            30,    // min window size
            1000,  // max window size
            100,   // fixed window size
            0.05,  // fixed threshold
            false, // hybrid mode off by default
        )
    }
}
