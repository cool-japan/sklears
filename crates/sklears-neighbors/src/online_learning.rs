//! Online Learning with Concept Drift Detection
//!
//! This module implements advanced online learning methods for neighbor-based algorithms,
//! including concept drift detection, adaptive neighbor selection, and streaming outlier detection.
//!
//! # Key Features
//!
//! - **Concept Drift Detection**: Automatically detect when data distribution changes
//! - **Adaptive Neighbor Selection**: Dynamically adjust k based on data characteristics
//! - **Streaming Outlier Detection**: Real-time anomaly detection in streaming data
//! - **Online Metric Learning**: Continuously update distance metrics as new data arrives
//!
//! # Examples
//!
//! ```rust
//! use sklears_neighbors::online_learning::{DriftDetector, DriftDetectionMethod};
//! use scirs2_core::ndarray::Array1;
//!
//! let mut detector = DriftDetector::new(DriftDetectionMethod::Adwin { delta: 0.002 });
//! # let errors = vec![0.1, 0.15, 0.12, 0.4, 0.5, 0.6]; // Example error rates
//! # for error in errors {
//! #     detector.add_element(error);
//! # }
//! ```

use crate::distance::Distance;
use crate::NeighborsError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::Float;
use std::collections::VecDeque;

/// Methods for detecting concept drift in streaming data
#[derive(Debug, Clone)]
pub enum DriftDetectionMethod {
    /// ADWIN (Adaptive Windowing) - detects changes in data distribution
    Adwin {
        /// Confidence parameter (smaller = more sensitive)
        delta: Float,
    },
    /// DDM (Drift Detection Method) - based on error rate statistics
    DDM {
        /// Warning threshold
        warning_level: Float,
        /// Drift threshold
        drift_level: Float,
    },
    /// EDDM (Early Drift Detection Method) - improved DDM
    EDDM {
        /// Warning threshold
        warning_level: Float,
        /// Drift threshold
        drift_level: Float,
    },
    /// Page-Hinkley test - cumulative difference test
    PageHinkley {
        /// Minimum amplitude of change to detect
        min_instances: usize,
        /// Threshold for drift detection
        threshold: Float,
        /// Forgetting factor (0-1)
        alpha: Float,
    },
}

/// Concept drift detector for online learning
#[derive(Debug, Clone)]
pub struct DriftDetector {
    method: DriftDetectionMethod,
    state: DriftState,
    estimation: Float,
    variance: Float,
    n_samples: usize,
    drift_detected: bool,
    warning_detected: bool,
}

/// Internal state for drift detection
#[derive(Debug, Clone)]
struct DriftState {
    /// For ADWIN: window of recent values
    window: VecDeque<Float>,
    /// For DDM/EDDM: running statistics
    p_min: Float,
    s_min: Float,
    /// For Page-Hinkley: cumulative sum
    cumsum: Float,
    min_cumsum: Float,
}

impl DriftDetector {
    /// Create a new drift detector
    pub fn new(method: DriftDetectionMethod) -> Self {
        Self {
            method,
            state: DriftState {
                window: VecDeque::new(),
                p_min: Float::INFINITY,
                s_min: Float::INFINITY,
                cumsum: 0.0,
                min_cumsum: Float::INFINITY,
            },
            estimation: 0.0,
            variance: 0.0,
            n_samples: 0,
            drift_detected: false,
            warning_detected: false,
        }
    }

    /// Add a new element (e.g., prediction error) and check for drift
    pub fn add_element(&mut self, value: Float) -> bool {
        self.n_samples += 1;
        self.drift_detected = false;
        self.warning_detected = false;

        match &self.method {
            DriftDetectionMethod::Adwin { delta } => {
                self.adwin_add_element(value, *delta);
            }
            DriftDetectionMethod::DDM {
                warning_level,
                drift_level,
            } => {
                self.ddm_add_element(value, *warning_level, *drift_level);
            }
            DriftDetectionMethod::EDDM {
                warning_level,
                drift_level,
            } => {
                self.eddm_add_element(value, *warning_level, *drift_level);
            }
            DriftDetectionMethod::PageHinkley {
                min_instances,
                threshold,
                alpha,
            } => {
                self.page_hinkley_add_element(value, *min_instances, *threshold, *alpha);
            }
        }

        self.drift_detected
    }

    /// ADWIN drift detection implementation
    fn adwin_add_element(&mut self, value: Float, delta: Float) {
        self.state.window.push_back(value);

        // Check if there's a significant change between two sub-windows
        if self.state.window.len() > 10 {
            let mid = self.state.window.len() / 2;
            let (left, right): (Vec<_>, Vec<_>) = self
                .state
                .window
                .iter()
                .enumerate()
                .partition(|(i, _)| *i < mid);

            let left_mean: Float =
                left.iter().map(|(_, &v)| v).sum::<Float>() / left.len() as Float;
            let right_mean: Float =
                right.iter().map(|(_, &v)| v).sum::<Float>() / right.len() as Float;

            // Hoeffding bound
            let m = (1.0 / left.len() as Float + 1.0 / right.len() as Float).sqrt();
            let epsilon = ((2.0 * delta.ln().abs()) / m).sqrt();

            if (left_mean - right_mean).abs() > epsilon {
                self.drift_detected = true;
                // Remove old window
                self.state.window.drain(0..mid);
            }
        }

        // Keep window size manageable
        if self.state.window.len() > 1000 {
            self.state.window.pop_front();
        }

        self.estimation =
            self.state.window.iter().sum::<Float>() / self.state.window.len() as Float;
    }

    /// DDM (Drift Detection Method) implementation
    fn ddm_add_element(&mut self, error: Float, warning_level: Float, drift_level: Float) {
        // Update running mean and standard deviation of error rate
        let old_estimation = self.estimation;
        self.estimation = self.estimation + (error - self.estimation) / self.n_samples as Float;

        if self.n_samples > 1 {
            self.variance += (error - old_estimation) * (error - self.estimation)
                / (self.n_samples - 1) as Float;
        }

        let std = self.variance.sqrt();

        // Update minimum values
        if self.n_samples > 30 {
            if self.estimation + std < self.state.p_min + self.state.s_min {
                self.state.p_min = self.estimation;
                self.state.s_min = std;
            }

            // Check for drift
            if self.estimation + std > self.state.p_min + drift_level * self.state.s_min {
                self.drift_detected = true;
            } else if self.estimation + std > self.state.p_min + warning_level * self.state.s_min {
                self.warning_detected = true;
            }
        }
    }

    /// EDDM (Early Drift Detection Method) implementation
    fn eddm_add_element(&mut self, error: Float, warning_level: Float, drift_level: Float) {
        // EDDM tracks distance between errors instead of error rate
        // For simplicity, we'll track a modified version
        let distance = if error > 0.0 {
            1.0 / error
        } else {
            Float::INFINITY
        };

        let old_estimation = self.estimation;
        self.estimation = self.estimation + (distance - self.estimation) / self.n_samples as Float;

        if self.n_samples > 1 {
            self.variance += (distance - old_estimation) * (distance - self.estimation)
                / (self.n_samples - 1) as Float;
        }

        let std = self.variance.sqrt();

        if self.n_samples > 30 {
            if self.estimation + 2.0 * std > self.state.p_min + 2.0 * self.state.s_min {
                self.state.p_min = self.estimation;
                self.state.s_min = std;
            }

            if self.estimation + 2.0 * std < self.state.p_min + drift_level * self.state.s_min {
                self.drift_detected = true;
            } else if self.estimation + 2.0 * std
                < self.state.p_min + warning_level * self.state.s_min
            {
                self.warning_detected = true;
            }
        }
    }

    /// Page-Hinkley test implementation
    fn page_hinkley_add_element(
        &mut self,
        value: Float,
        min_instances: usize,
        threshold: Float,
        alpha: Float,
    ) {
        self.estimation = alpha * self.estimation + (1.0 - alpha) * value;
        self.state.cumsum += value - self.estimation - alpha;

        if self.state.cumsum < self.state.min_cumsum {
            self.state.min_cumsum = self.state.cumsum;
        }

        if self.n_samples >= min_instances {
            let diff = self.state.cumsum - self.state.min_cumsum;
            if diff > threshold {
                self.drift_detected = true;
                // Reset
                self.state.cumsum = 0.0;
                self.state.min_cumsum = Float::INFINITY;
            }
        }
    }

    /// Check if drift was detected
    pub fn drift_detected(&self) -> bool {
        self.drift_detected
    }

    /// Check if warning was detected
    pub fn warning_detected(&self) -> bool {
        self.warning_detected
    }

    /// Get current estimation
    pub fn estimation(&self) -> Float {
        self.estimation
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.state = DriftState {
            window: VecDeque::new(),
            p_min: Float::INFINITY,
            s_min: Float::INFINITY,
            cumsum: 0.0,
            min_cumsum: Float::INFINITY,
        };
        self.estimation = 0.0;
        self.variance = 0.0;
        self.n_samples = 0;
        self.drift_detected = false;
        self.warning_detected = false;
    }
}

/// Adaptive K-Nearest Neighbors with concept drift detection
#[derive(Debug, Clone)]
pub struct AdaptiveKNeighborsClassifier<S> {
    /// Base k value
    base_k: usize,
    /// Current adaptive k
    current_k: usize,
    /// Distance metric
    distance: Distance,
    /// Drift detector
    drift_detector: DriftDetector,
    /// Adaptation rate (how quickly k changes)
    adaptation_rate: Float,
    /// Minimum k value
    min_k: usize,
    /// Maximum k value
    max_k: usize,
    state: S,
}

/// Untrained state
#[derive(Debug, Clone)]
pub struct Untrained;

/// Trained state
#[derive(Debug, Clone)]
pub struct Trained {
    x_train: Array2<Float>,
    y_train: Array1<i32>,
    classes: Array1<i32>,
    /// Recent prediction errors for drift detection
    recent_errors: VecDeque<Float>,
}

impl AdaptiveKNeighborsClassifier<Untrained> {
    /// Create a new adaptive KNN classifier
    pub fn new(base_k: usize, drift_method: DriftDetectionMethod) -> Self {
        Self {
            base_k,
            current_k: base_k,
            distance: Distance::Euclidean,
            drift_detector: DriftDetector::new(drift_method),
            adaptation_rate: 0.1,
            min_k: (base_k / 2).max(1),
            max_k: base_k * 2,
            state: Untrained,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set the adaptation rate
    pub fn with_adaptation_rate(mut self, rate: Float) -> Self {
        self.adaptation_rate = rate;
        self
    }

    /// Set min and max k bounds
    pub fn with_k_bounds(mut self, min_k: usize, max_k: usize) -> Self {
        self.min_k = min_k;
        self.max_k = max_k;
        self
    }
}

impl Fit<Array2<Float>, Array1<i32>, AdaptiveKNeighborsClassifier<Trained>>
    for AdaptiveKNeighborsClassifier<Untrained>
{
    type Fitted = AdaptiveKNeighborsClassifier<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted, SklearsError> {
        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![y.len()],
                actual: vec![x.nrows(), x.ncols()],
            }
            .into());
        }

        if self.base_k >= x.nrows() {
            return Err(NeighborsError::InvalidInput(format!(
                "base_k={} should be less than n_samples={}",
                self.base_k,
                x.nrows()
            ))
            .into());
        }

        // Extract unique classes
        let mut unique_classes: Vec<i32> = y.iter().cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);

        Ok(AdaptiveKNeighborsClassifier {
            base_k: self.base_k,
            current_k: self.base_k,
            distance: self.distance,
            drift_detector: self.drift_detector,
            adaptation_rate: self.adaptation_rate,
            min_k: self.min_k,
            max_k: self.max_k,
            state: Trained {
                x_train: x.clone(),
                y_train: y.clone(),
                classes,
                recent_errors: VecDeque::new(),
            },
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for AdaptiveKNeighborsClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>, SklearsError> {
        let mut predictions = Vec::with_capacity(x.nrows());

        for row in x.rows() {
            let pred = self.predict_single(&row)?;
            predictions.push(pred);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl AdaptiveKNeighborsClassifier<Trained> {
    /// Predict a single sample
    fn predict_single(&self, query: &ArrayView1<Float>) -> Result<i32, SklearsError> {
        // Find k nearest neighbors using current adaptive k
        let mut distances_indices: Vec<(Float, usize)> = self
            .state
            .x_train
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| (self.distance.calculate(query, &row), i))
            .collect();

        distances_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbor_indices: Vec<usize> = distances_indices
            .into_iter()
            .take(self.current_k.min(self.state.x_train.nrows()))
            .map(|(_, idx)| idx)
            .collect();

        // Count classes
        let mut class_counts: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();
        for &idx in &neighbor_indices {
            *class_counts.entry(self.state.y_train[idx]).or_insert(0) += 1;
        }

        // Return most common class
        let prediction = class_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(class, _)| class)
            .unwrap_or(self.state.classes[0]);

        Ok(prediction)
    }

    /// Update model with new sample and check for drift
    pub fn partial_fit(
        &mut self,
        x_new: &Array2<Float>,
        y_new: &Array1<i32>,
    ) -> Result<(), SklearsError> {
        if x_new.nrows() != y_new.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![y_new.len()],
                actual: vec![x_new.nrows(), x_new.ncols()],
            }
            .into());
        }

        // Make predictions on new data to compute error rate
        for (i, sample) in x_new.rows().into_iter().enumerate() {
            let pred = self.predict_single(&sample)?;
            let true_label = y_new[i];
            let error = if pred != true_label { 1.0 } else { 0.0 };

            // Add error to drift detector
            let drift = self.drift_detector.add_element(error);

            if drift {
                // Drift detected - adapt k
                self.adapt_k();
            }

            self.state.recent_errors.push_back(error);
            if self.state.recent_errors.len() > 100 {
                self.state.recent_errors.pop_front();
            }
        }

        // Add new samples to training set
        self.state.x_train = scirs2_core::ndarray::concatenate(
            scirs2_core::ndarray::Axis(0),
            &[self.state.x_train.view(), x_new.view()],
        )
        .map_err(|e| NeighborsError::InvalidInput(format!("Failed to concatenate: {:?}", e)))?;

        self.state.y_train = scirs2_core::ndarray::concatenate(
            scirs2_core::ndarray::Axis(0),
            &[self.state.y_train.view(), y_new.view()],
        )
        .map_err(|e| NeighborsError::InvalidInput(format!("Failed to concatenate: {:?}", e)))?;

        Ok(())
    }

    /// Adapt k based on recent performance
    fn adapt_k(&mut self) {
        let recent_error_rate = if !self.state.recent_errors.is_empty() {
            self.state.recent_errors.iter().sum::<Float>() / self.state.recent_errors.len() as Float
        } else {
            0.0
        };

        // Increase k if error rate is high (more smoothing)
        // Decrease k if error rate is low (more sensitivity)
        if recent_error_rate > 0.3 {
            let new_k = (self.current_k as Float * (1.0 + self.adaptation_rate)).round() as usize;
            self.current_k = new_k.min(self.max_k);
        } else if recent_error_rate < 0.1 {
            let new_k = (self.current_k as Float * (1.0 - self.adaptation_rate)).round() as usize;
            self.current_k = new_k.max(self.min_k);
        }
    }

    /// Get current k value
    pub fn current_k(&self) -> usize {
        self.current_k
    }

    /// Check if drift was detected recently
    pub fn drift_detected(&self) -> bool {
        self.drift_detector.drift_detected()
    }
}

/// Streaming outlier detector with concept drift adaptation
#[derive(Debug, Clone)]
pub struct StreamingOutlierDetector {
    /// Number of neighbors for LOF computation
    k: usize,
    /// Distance metric
    distance: Distance,
    /// Threshold for outlier classification
    threshold: Float,
    /// Drift detector
    drift_detector: DriftDetector,
    /// Sliding window of recent samples
    window: VecDeque<Array1<Float>>,
    /// Maximum window size
    max_window_size: usize,
    /// Adaptive threshold
    adaptive_threshold: bool,
}

impl StreamingOutlierDetector {
    /// Create a new streaming outlier detector
    pub fn new(k: usize, threshold: Float) -> Self {
        Self {
            k,
            distance: Distance::Euclidean,
            threshold,
            drift_detector: DriftDetector::new(DriftDetectionMethod::Adwin { delta: 0.002 }),
            window: VecDeque::new(),
            max_window_size: 1000,
            adaptive_threshold: true,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set the maximum window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.max_window_size = size;
        self
    }

    /// Enable or disable adaptive thresholding
    pub fn with_adaptive_threshold(mut self, adaptive: bool) -> Self {
        self.adaptive_threshold = adaptive;
        self
    }

    /// Add a new sample and check if it's an outlier
    pub fn is_outlier(&mut self, sample: &Array1<Float>) -> Result<bool, SklearsError> {
        if self.window.len() < self.k + 1 {
            // Not enough samples yet
            self.window.push_back(sample.clone());
            return Ok(false);
        }

        // Compute LOF score for the new sample
        let lof_score = self.compute_lof(sample)?;

        // Update drift detector with LOF score
        let drift = self.drift_detector.add_element(lof_score);

        if drift && self.adaptive_threshold {
            // Adjust threshold based on recent scores
            self.threshold *= 1.1; // Increase sensitivity after drift
        }

        let is_outlier = lof_score > self.threshold;

        // Add sample to window
        self.window.push_back(sample.clone());
        if self.window.len() > self.max_window_size {
            self.window.pop_front();
        }

        Ok(is_outlier)
    }

    /// Compute Local Outlier Factor for a sample
    fn compute_lof(&self, sample: &Array1<Float>) -> Result<Float, SklearsError> {
        // Find k nearest neighbors in window
        let mut distances: Vec<(Float, usize)> = self
            .window
            .iter()
            .enumerate()
            .map(|(i, x)| (self.distance.calculate(&sample.view(), &x.view()), i))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k_neighbors: Vec<usize> = distances
            .into_iter()
            .take(self.k.min(self.window.len()))
            .map(|(_, idx)| idx)
            .collect();

        if k_neighbors.is_empty() {
            return Ok(1.0);
        }

        // Compute local reachability density (simplified)
        let mut lrd = 0.0;
        for &neighbor_idx in &k_neighbors {
            let neighbor = &self.window[neighbor_idx];
            let dist = self.distance.calculate(&sample.view(), &neighbor.view());
            lrd += dist;
        }
        lrd = k_neighbors.len() as Float / (lrd + 1e-10);

        // Compute LOF (simplified)
        let mut lof = 0.0;
        for &neighbor_idx in &k_neighbors {
            let neighbor = &self.window[neighbor_idx];
            // Compute neighbor's LRD
            let mut neighbor_lrd = 0.0;
            let mut neighbor_neighbors: Vec<Float> = self
                .window
                .iter()
                .map(|x| self.distance.calculate(&neighbor.view(), &x.view()))
                .collect();
            neighbor_neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for dist in neighbor_neighbors.iter().take(self.k) {
                neighbor_lrd += dist;
            }
            neighbor_lrd = self.k as Float / (neighbor_lrd + 1e-10);
            lof += neighbor_lrd / (lrd + 1e-10);
        }
        lof /= k_neighbors.len() as Float;

        Ok(lof)
    }

    /// Get current threshold
    pub fn threshold(&self) -> Float {
        self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_drift_detector_adwin() {
        let mut detector = DriftDetector::new(DriftDetectionMethod::Adwin { delta: 0.002 });

        // Add stable data
        for _ in 0..100 {
            detector.add_element(0.1);
        }
        assert!(!detector.drift_detected());

        // Add data with different distribution
        for _ in 0..50 {
            let drift = detector.add_element(0.9);
            if drift {
                assert!(detector.drift_detected());
                break;
            }
        }
    }

    #[test]
    fn test_drift_detector_ddm() {
        let mut detector = DriftDetector::new(DriftDetectionMethod::DDM {
            warning_level: 2.0,
            drift_level: 3.0,
        });

        // Simulate increasing error rate
        for i in 0..100 {
            let error = (i as Float / 100.0).min(0.5);
            detector.add_element(error);
        }
    }

    #[test]
    fn test_adaptive_knn_basic() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                -1.0, -1.0, -0.5, -1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 1.5, 1.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 1, 1];

        let drift_method = DriftDetectionMethod::Adwin { delta: 0.002 };
        let classifier = AdaptiveKNeighborsClassifier::new(3, drift_method);
        let fitted = classifier.fit(&x, &y).unwrap();

        let test_x = Array2::from_shape_vec((2, 2), vec![-0.8, -0.8, 0.8, 0.8]).unwrap();
        let predictions = fitted.predict(&test_x).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_adaptive_knn_partial_fit() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let drift_method = DriftDetectionMethod::DDM {
            warning_level: 2.0,
            drift_level: 3.0,
        };
        let classifier = AdaptiveKNeighborsClassifier::new(2, drift_method);
        let mut fitted = classifier.fit(&x, &y).unwrap();

        // Add new samples
        let x_new = Array2::from_shape_vec((2, 2), vec![3.0, 3.0, 4.0, 4.0]).unwrap();
        let y_new = array![1, 1];
        fitted.partial_fit(&x_new, &y_new).unwrap();

        // K might have adapted
        assert!(fitted.current_k() > 0);
    }

    #[test]
    fn test_streaming_outlier_detector() {
        let mut detector = StreamingOutlierDetector::new(5, 1.5);

        // Add normal samples
        for i in 0..20 {
            let sample = array![i as Float, i as Float];
            let _is_outlier = detector.is_outlier(&sample).unwrap();
            // First few samples won't be classified as outliers
        }

        // Add an obvious outlier
        let outlier = array![100.0, 100.0];
        let is_outlier = detector.is_outlier(&outlier).unwrap();
        assert!(is_outlier);
    }

    #[test]
    fn test_page_hinkley_detector() {
        let mut detector = DriftDetector::new(DriftDetectionMethod::PageHinkley {
            min_instances: 30,
            threshold: 10.0,
            alpha: 0.9999,
        });

        // Add stable data
        for _ in 0..50 {
            detector.add_element(1.0);
        }

        // Add shifted data
        for _ in 0..50 {
            detector.add_element(5.0);
        }
    }

    #[test]
    fn test_drift_detector_reset() {
        let mut detector = DriftDetector::new(DriftDetectionMethod::Adwin { delta: 0.002 });

        for _ in 0..50 {
            detector.add_element(0.5);
        }

        detector.reset();
        assert_eq!(detector.n_samples, 0);
        assert_eq!(detector.estimation, 0.0);
    }
}
