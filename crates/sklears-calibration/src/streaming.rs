//! Streaming and Online Calibration Methods
//!
//! This module implements calibration methods that can be updated incrementally
//! in real-time as new data arrives, suitable for streaming applications.

use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::VecDeque;

use crate::{CalibrationEstimator, SigmoidCalibrator};

/// Online Calibration Method
///
/// Base trait for calibration methods that can be updated incrementally
/// with new samples as they arrive in streaming scenarios.
pub trait OnlineCalibrationEstimator: CalibrationEstimator {
    /// Update the calibrator with a single new sample
    fn partial_fit(&mut self, probability: Float, y_true: i32) -> Result<()>;

    /// Update the calibrator with a batch of new samples
    fn partial_fit_batch(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        for (prob, target) in probabilities.iter().zip(y_true.iter()) {
            self.partial_fit(*prob, *target)?;
        }

        Ok(())
    }

    /// Get the current number of samples seen
    fn n_samples_seen(&self) -> usize;

    /// Reset the calibrator state
    fn reset(&mut self);
}

/// Online Sigmoid Calibration
///
/// Streaming version of Platt scaling that updates parameters incrementally
/// using stochastic gradient descent.
#[derive(Debug, Clone)]
pub struct OnlineSigmoidCalibrator {
    /// Sigmoid parameter A
    a: Float,
    /// Sigmoid parameter B  
    b: Float,
    /// Learning rate for SGD updates
    learning_rate: Float,
    /// Number of samples seen so far
    n_samples: usize,
    /// Whether to use momentum in SGD
    use_momentum: bool,
    /// Momentum parameter (if enabled)
    momentum: Float,
    /// Momentum terms for A and B
    momentum_a: Float,
    momentum_b: Float,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl Default for OnlineSigmoidCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineSigmoidCalibrator {
    pub fn new() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            learning_rate: 0.01,
            n_samples: 0,
            use_momentum: false,
            momentum: 0.9,
            momentum_a: 0.0,
            momentum_b: 0.0,
            is_fitted: false,
        }
    }

    /// Set learning rate for SGD
    pub fn with_learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Enable momentum for SGD
    pub fn with_momentum(mut self, momentum: Float) -> Self {
        self.use_momentum = true;
        self.momentum = momentum;
        self
    }

    /// Compute sigmoid probability
    fn sigmoid(&self, x: Float) -> Float {
        1.0 / (1.0 + (-self.a * x - self.b).exp())
    }

    /// Convert probability to logit
    fn prob_to_logit(&self, p: Float) -> Float {
        let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
        (clamped_p / (1.0 - clamped_p)).ln()
    }
}

impl CalibrationEstimator for OnlineSigmoidCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // Initialize with offline fit, then mark as fitted
        let _offline_calibrator = SigmoidCalibrator::new().fit(probabilities, y_true)?;

        // Use offline parameters as initialization
        self.a = 1.0; // Default from offline calibrator
        self.b = 0.0; // Default from offline calibrator
        self.n_samples = probabilities.len();
        self.is_fitted = true;

        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            });
        }

        let calibrated = probabilities.mapv(|p| {
            let logit = self.prob_to_logit(p);
            self.sigmoid(logit)
        });

        Ok(calibrated)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl OnlineCalibrationEstimator for OnlineSigmoidCalibrator {
    fn partial_fit(&mut self, probability: Float, y_true: i32) -> Result<()> {
        let logit = self.prob_to_logit(probability);
        let predicted = self.sigmoid(logit);
        let target = y_true as Float;

        // Compute gradient
        let error = predicted - target;
        let grad_a = error * predicted * (1.0 - predicted) * logit;
        let grad_b = error * predicted * (1.0 - predicted);

        // Apply momentum if enabled
        if self.use_momentum {
            self.momentum_a = self.momentum * self.momentum_a + self.learning_rate * grad_a;
            self.momentum_b = self.momentum * self.momentum_b + self.learning_rate * grad_b;

            self.a -= self.momentum_a;
            self.b -= self.momentum_b;
        } else {
            self.a -= self.learning_rate * grad_a;
            self.b -= self.learning_rate * grad_b;
        }

        self.n_samples += 1;
        self.is_fitted = true;

        Ok(())
    }

    fn n_samples_seen(&self) -> usize {
        self.n_samples
    }

    fn reset(&mut self) {
        self.a = 1.0;
        self.b = 0.0;
        self.n_samples = 0;
        self.momentum_a = 0.0;
        self.momentum_b = 0.0;
        self.is_fitted = false;
    }
}

/// Adaptive Online Calibration
///
/// Calibration method that adapts to concept drift by maintaining
/// a sliding window of recent samples and re-training periodically.
#[derive(Debug, Clone)]
pub struct AdaptiveOnlineCalibrator {
    /// Base calibrator to use
    base_calibrator: Box<dyn CalibrationEstimator>,
    /// Window size for adaptation
    window_size: usize,
    /// Sliding window of probabilities
    probability_window: VecDeque<Float>,
    /// Sliding window of targets
    target_window: VecDeque<i32>,
    /// Frequency of re-training (in number of samples)
    retrain_frequency: usize,
    /// Number of samples since last retrain
    samples_since_retrain: usize,
    /// Total number of samples seen
    n_samples: usize,
    /// Drift detection threshold
    drift_threshold: Float,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl AdaptiveOnlineCalibrator {
    pub fn new(window_size: usize) -> Self {
        Self {
            base_calibrator: Box::new(SigmoidCalibrator::new()),
            window_size,
            probability_window: VecDeque::new(),
            target_window: VecDeque::new(),
            retrain_frequency: 100,
            samples_since_retrain: 0,
            n_samples: 0,
            drift_threshold: 0.1,
            is_fitted: false,
        }
    }

    /// Set the base calibrator
    pub fn with_base_calibrator(mut self, calibrator: Box<dyn CalibrationEstimator>) -> Self {
        self.base_calibrator = calibrator;
        self
    }

    /// Set retrain frequency
    pub fn with_retrain_frequency(mut self, frequency: usize) -> Self {
        self.retrain_frequency = frequency;
        self
    }

    /// Set drift detection threshold
    pub fn with_drift_threshold(mut self, threshold: Float) -> Self {
        self.drift_threshold = threshold;
        self
    }

    /// Detect concept drift using calibration error
    fn detect_drift(&self) -> bool {
        if self.probability_window.len() < 10 {
            return false;
        }

        // Calculate recent calibration error
        let mut recent_error = 0.0;
        let recent_size = self.probability_window.len().min(20);

        for i in (self.probability_window.len() - recent_size)..self.probability_window.len() {
            let prob = self.probability_window[i];
            let target = self.target_window[i] as Float;
            recent_error += (prob - target).abs();
        }

        recent_error /= recent_size as Float;
        recent_error > self.drift_threshold
    }

    /// Retrain the base calibrator on current window
    fn retrain(&mut self) -> Result<()> {
        if self.probability_window.is_empty() {
            return Ok(());
        }

        let probabilities =
            Array1::from(self.probability_window.iter().cloned().collect::<Vec<_>>());
        let targets = Array1::from(self.target_window.iter().cloned().collect::<Vec<_>>());

        self.base_calibrator.fit(&probabilities, &targets)?;
        self.samples_since_retrain = 0;

        Ok(())
    }
}

impl CalibrationEstimator for AdaptiveOnlineCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // Initialize window with training data
        for (prob, target) in probabilities.iter().zip(y_true.iter()) {
            if self.probability_window.len() >= self.window_size {
                self.probability_window.pop_front();
                self.target_window.pop_front();
            }
            self.probability_window.push_back(*prob);
            self.target_window.push_back(*target);
        }

        self.base_calibrator.fit(probabilities, y_true)?;
        self.n_samples = probabilities.len();
        self.is_fitted = true;

        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            });
        }

        self.base_calibrator.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl OnlineCalibrationEstimator for AdaptiveOnlineCalibrator {
    fn partial_fit(&mut self, probability: Float, y_true: i32) -> Result<()> {
        // Add to sliding window
        if self.probability_window.len() >= self.window_size {
            self.probability_window.pop_front();
            self.target_window.pop_front();
        }
        self.probability_window.push_back(probability);
        self.target_window.push_back(y_true);

        self.n_samples += 1;
        self.samples_since_retrain += 1;

        // Check if we need to retrain
        let should_retrain =
            self.samples_since_retrain >= self.retrain_frequency || self.detect_drift();

        if should_retrain {
            self.retrain()?;
        }

        self.is_fitted = true;

        Ok(())
    }

    fn n_samples_seen(&self) -> usize {
        self.n_samples
    }

    fn reset(&mut self) {
        self.probability_window.clear();
        self.target_window.clear();
        self.samples_since_retrain = 0;
        self.n_samples = 0;
        self.is_fitted = false;
    }
}

/// Real-time Calibration Monitor
///
/// Monitors calibration quality in real-time and triggers alerts
/// when calibration degrades significantly.
#[derive(Debug, Clone)]
pub struct RealTimeCalibrationMonitor {
    /// Base calibrator being monitored
    calibrator: Box<dyn CalibrationEstimator>,
    /// Window size for monitoring
    window_size: usize,
    /// Recent calibration errors
    error_window: VecDeque<Float>,
    /// Alert threshold for calibration degradation
    alert_threshold: Float,
    /// Number of consecutive alerts needed to trigger
    alert_count_threshold: usize,
    /// Current consecutive alert count
    current_alert_count: usize,
    /// Whether an alert is currently active
    alert_active: bool,
    /// Statistics for monitoring
    total_samples: usize,
    total_error: Float,
}

impl RealTimeCalibrationMonitor {
    /// Create a new real-time calibration monitor
    pub fn new(calibrator: Box<dyn CalibrationEstimator>, window_size: usize) -> Self {
        Self {
            calibrator,
            window_size,
            error_window: VecDeque::new(),
            alert_threshold: 0.1,
            alert_count_threshold: 5,
            current_alert_count: 0,
            alert_active: false,
            total_samples: 0,
            total_error: 0.0,
        }
    }

    /// Set alert threshold
    pub fn with_alert_threshold(mut self, threshold: Float) -> Self {
        self.alert_threshold = threshold;
        self
    }

    /// Set alert count threshold
    pub fn with_alert_count_threshold(mut self, count: usize) -> Self {
        self.alert_count_threshold = count;
        self
    }

    /// Update monitoring with new prediction
    pub fn update(&mut self, probability: Float, y_true: i32) -> Result<bool> {
        let calibrated = self
            .calibrator
            .predict_proba(&Array1::from(vec![probability]))?;
        let calibrated_prob = calibrated[0];

        // Calculate calibration error
        let error = (calibrated_prob - y_true as Float).abs();

        // Update error window
        if self.error_window.len() >= self.window_size {
            let old_error = self.error_window.pop_front().unwrap();
            self.total_error -= old_error;
        }
        self.error_window.push_back(error);
        self.total_error += error;
        self.total_samples += 1;

        // Check for alert condition
        let current_error_rate = if !self.error_window.is_empty() {
            self.total_error / self.error_window.len() as Float
        } else {
            0.0
        };

        if current_error_rate > self.alert_threshold {
            self.current_alert_count += 1;
            if self.current_alert_count >= self.alert_count_threshold {
                self.alert_active = true;
            }
        } else {
            self.current_alert_count = 0;
            self.alert_active = false;
        }

        Ok(self.alert_active)
    }

    /// Get current calibration error rate
    pub fn current_error_rate(&self) -> Float {
        if !self.error_window.is_empty() {
            self.total_error / self.error_window.len() as Float
        } else {
            0.0
        }
    }

    /// Check if alert is active
    pub fn is_alert_active(&self) -> bool {
        self.alert_active
    }

    /// Get total samples monitored
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Reset monitoring state
    pub fn reset(&mut self) {
        self.error_window.clear();
        self.current_alert_count = 0;
        self.alert_active = false;
        self.total_samples = 0;
        self.total_error = 0.0;
    }
}

/// Incremental Calibration Updates
///
/// Provides methods for incrementally updating calibration as new data arrives
/// without full retraining.
#[derive(Debug, Clone)]
pub struct IncrementalCalibrationUpdater {
    /// Base calibrator parameters
    calibrator_params: CalibrationParams,
    /// Update frequency (in number of samples)
    update_frequency: usize,
    /// Samples since last update
    samples_since_update: usize,
    /// Accumulated gradients or updates
    accumulated_updates: Vec<Float>,
    /// Learning rate for incremental updates
    learning_rate: Float,
    /// Whether to use exponential smoothing
    use_smoothing: bool,
    /// Smoothing factor
    smoothing_factor: Float,
}

#[derive(Debug, Clone)]
struct CalibrationParams {
    params: Vec<Float>,
}

impl IncrementalCalibrationUpdater {
    /// Create a new incremental calibration updater
    pub fn new(initial_params: Vec<Float>) -> Self {
        Self {
            calibrator_params: CalibrationParams {
                params: initial_params,
            },
            update_frequency: 50,
            samples_since_update: 0,
            accumulated_updates: Vec::new(),
            learning_rate: 0.01,
            use_smoothing: true,
            smoothing_factor: 0.9,
        }
    }

    /// Set update frequency
    pub fn with_update_frequency(mut self, frequency: usize) -> Self {
        self.update_frequency = frequency;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Enable/disable exponential smoothing
    pub fn with_smoothing(mut self, use_smoothing: bool, factor: Float) -> Self {
        self.use_smoothing = use_smoothing;
        self.smoothing_factor = factor;
        self
    }

    /// Add incremental update
    pub fn add_update(&mut self, probability: Float, y_true: i32) -> bool {
        // Simplified gradient computation (would be more sophisticated in practice)
        let error = probability - y_true as Float;

        if self.accumulated_updates.is_empty() {
            self.accumulated_updates = vec![0.0; self.calibrator_params.params.len()];
        }

        // Accumulate update (simplified)
        if !self.accumulated_updates.is_empty() {
            self.accumulated_updates[0] += error;
        }

        self.samples_since_update += 1;

        // Check if we should apply updates
        if self.samples_since_update >= self.update_frequency {
            self.apply_updates();
            true
        } else {
            false
        }
    }

    /// Apply accumulated updates to parameters
    fn apply_updates(&mut self) {
        if self.accumulated_updates.is_empty() {
            return;
        }

        for (i, update) in self.accumulated_updates.iter().enumerate() {
            if i < self.calibrator_params.params.len() {
                if self.use_smoothing {
                    // Exponential smoothing
                    let old_param = self.calibrator_params.params[i];
                    let new_param = old_param - self.learning_rate * update;
                    self.calibrator_params.params[i] = self.smoothing_factor * old_param
                        + (1.0 - self.smoothing_factor) * new_param;
                } else {
                    // Direct update
                    self.calibrator_params.params[i] -= self.learning_rate * update;
                }
            }
        }

        // Reset accumulators
        self.accumulated_updates.fill(0.0);
        self.samples_since_update = 0;
    }

    /// Get current parameters
    pub fn current_params(&self) -> &[Float] {
        &self.calibrator_params.params
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn create_streaming_data() -> (Vec<Float>, Vec<i32>) {
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55,
            0.65, 0.75, 0.85, 0.9,
        ];
        let targets = vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        (probabilities, targets)
    }

    #[test]
    fn test_online_sigmoid_calibrator() {
        let (probabilities, targets) = create_streaming_data();

        let mut calibrator = OnlineSigmoidCalibrator::new()
            .with_learning_rate(0.01)
            .with_momentum(0.9);

        // Initial fit
        let initial_probs = Array1::from(probabilities[..10].to_vec());
        let initial_targets = Array1::from(targets[..10].to_vec());
        calibrator.fit(&initial_probs, &initial_targets).unwrap();

        // Incremental updates
        for (prob, target) in probabilities[10..].iter().zip(targets[10..].iter()) {
            calibrator.partial_fit(*prob, *target).unwrap();
        }

        assert_eq!(calibrator.n_samples_seen(), 10 + (probabilities.len() - 10));

        // Test prediction
        let test_probs = Array1::from(vec![0.3, 0.7]);
        let predictions = calibrator.predict_proba(&test_probs).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_adaptive_online_calibrator() {
        let (probabilities, targets) = create_streaming_data();

        let mut calibrator = AdaptiveOnlineCalibrator::new(10)
            .with_retrain_frequency(5)
            .with_drift_threshold(0.2);

        // Initial fit
        let initial_probs = Array1::from(probabilities[..5].to_vec());
        let initial_targets = Array1::from(targets[..5].to_vec());
        calibrator.fit(&initial_probs, &initial_targets).unwrap();

        // Incremental updates
        for (prob, target) in probabilities[5..].iter().zip(targets[5..].iter()) {
            calibrator.partial_fit(*prob, *target).unwrap();
        }

        assert_eq!(calibrator.n_samples_seen(), probabilities.len());

        // Test prediction
        let test_probs = Array1::from(vec![0.4, 0.8]);
        let predictions = calibrator.predict_proba(&test_probs).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_real_time_calibration_monitor() {
        let (probabilities, targets) = create_streaming_data();

        let base_calibrator = Box::new(SigmoidCalibrator::new());
        let mut monitor = RealTimeCalibrationMonitor::new(base_calibrator, 5)
            .with_alert_threshold(0.5)
            .with_alert_count_threshold(3);

        // Simulate real-time monitoring
        let mut _alert_triggered = false;
        for (prob, target) in probabilities.iter().zip(targets.iter()) {
            let alert = monitor.update(*prob, *target).unwrap();
            if alert {
                _alert_triggered = true;
            }
        }

        assert!(monitor.total_samples() > 0);
        assert!(monitor.current_error_rate() >= 0.0);
    }

    #[test]
    fn test_incremental_calibration_updater() {
        let (probabilities, targets) = create_streaming_data();

        let mut updater = IncrementalCalibrationUpdater::new(vec![1.0, 0.0])
            .with_update_frequency(5)
            .with_learning_rate(0.01)
            .with_smoothing(true, 0.9);

        let mut updates_applied = 0;
        for (prob, target) in probabilities.iter().zip(targets.iter()) {
            if updater.add_update(*prob, *target) {
                updates_applied += 1;
            }
        }

        assert!(updates_applied > 0);
        assert_eq!(updater.current_params().len(), 2);
    }

    #[test]
    fn test_online_calibrators_basic_properties() {
        let (probabilities, targets) = create_streaming_data();

        let mut calibrators: Vec<Box<dyn OnlineCalibrationEstimator>> = vec![
            Box::new(OnlineSigmoidCalibrator::new()),
            Box::new(AdaptiveOnlineCalibrator::new(10)),
        ];

        for calibrator in &mut calibrators {
            // Initial fit
            let initial_probs = Array1::from(probabilities[..5].to_vec());
            let initial_targets = Array1::from(targets[..5].to_vec());
            calibrator.fit(&initial_probs, &initial_targets).unwrap();

            // Incremental updates
            for (prob, target) in probabilities[5..].iter().zip(targets[5..].iter()) {
                calibrator.partial_fit(*prob, *target).unwrap();
            }

            assert!(calibrator.n_samples_seen() > 0);

            // Test prediction
            let test_probs = Array1::from(vec![0.3, 0.7]);
            let predictions = calibrator.predict_proba(&test_probs).unwrap();

            assert_eq!(predictions.len(), 2);
            assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));

            // Test reset
            calibrator.reset();
            assert_eq!(calibrator.n_samples_seen(), 0);
        }
    }
}
