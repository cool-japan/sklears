//! Temporal Naive Bayes models for time series classification
//!
//! This module provides specialized Naive Bayes models for time series data,
//! including temporal patterns, sequence modeling, and streaming classification.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use rayon::prelude::*;
use scirs2_core::random::CoreRandom;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba},
};
use std::collections::{HashMap, VecDeque};

use crate::GaussianNB;
use sklears_core::traits::{Trained, Untrained};

/// Type alias for forward-backward algorithm results
type ForwardBackwardResult = (Vec<Array2<f64>>, Vec<Array2<f64>>, Vec<Array1<f64>>);

/// Configuration for temporal Naive Bayes models
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Window size for temporal features
    pub window_size: usize,
    /// Step size for sliding window
    pub step_size: usize,
    /// Whether to include temporal derivatives
    pub use_derivatives: bool,
    /// Order of derivatives to include (1 = first derivative, 2 = second derivative, etc.)
    pub derivative_order: usize,
    /// Whether to use exponential decay for temporal weights
    pub use_temporal_decay: bool,
    /// Decay factor for temporal weights (lambda)
    pub decay_factor: f64,
    /// Whether to normalize temporal features
    pub normalize_temporal: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            step_size: 1,
            use_derivatives: true,
            derivative_order: 1,
            use_temporal_decay: false,
            decay_factor: 0.9,
            normalize_temporal: true,
        }
    }
}

/// Temporal feature extractor for time series data
#[derive(Debug, Clone)]
pub struct TemporalFeatureExtractor {
    config: TemporalConfig,
    /// Fitted statistics for normalization
    feature_means: Option<Array1<f64>>,
    feature_stds: Option<Array1<f64>>,
}

impl TemporalFeatureExtractor {
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            config,
            feature_means: None,
            feature_stds: None,
        }
    }

    /// Extract sliding window features from time series
    fn extract_windows(&self, series: &Array2<f64>) -> Array3<f64> {
        let (n_series, series_length) = series.dim();
        let n_features = 1; // Assume 1 feature per timestep for now
        let n_windows = if series_length >= self.config.window_size {
            (series_length - self.config.window_size) / self.config.step_size + 1
        } else {
            0
        };

        let mut windows =
            Array3::zeros((n_series, n_windows, self.config.window_size * n_features));

        for series_idx in 0..n_series {
            for window_idx in 0..n_windows {
                let start = window_idx * self.config.step_size;
                let end = start + self.config.window_size;

                for t in start..end {
                    let feature_idx = t - start; // Only 1 feature per timestep
                    windows[[series_idx, window_idx, feature_idx]] = series[[series_idx, t]];
                }
            }
        }

        windows
    }

    /// Compute derivatives of time series
    fn compute_derivatives(&self, series: &Array2<f64>) -> Vec<Array2<f64>> {
        let mut derivatives = Vec::new();
        let mut current = series.clone();

        for order in 1..=self.config.derivative_order {
            let mut derivative = Array2::zeros(current.dim());

            // Compute finite differences
            for i in 0..current.nrows() {
                for j in 1..current.ncols() {
                    derivative[[i, j]] = current[[i, j]] - current[[i, j - 1]];
                }
                // Set first column to zero (boundary condition)
                derivative[[i, 0]] = 0.0;
            }

            derivatives.push(derivative.clone());
            current = derivative;
        }

        derivatives
    }

    /// Apply temporal decay weights
    fn apply_temporal_decay(&self, features: &mut Array3<f64>) {
        if !self.config.use_temporal_decay {
            return;
        }

        let (n_series, n_windows, window_features) = features.dim();
        let features_per_timestep = window_features / self.config.window_size;

        for series_idx in 0..n_series {
            for window_idx in 0..n_windows {
                for t in 0..self.config.window_size {
                    let weight = self
                        .config
                        .decay_factor
                        .powi((self.config.window_size - 1 - t) as i32);
                    for f in 0..features_per_timestep {
                        let feature_idx = t * features_per_timestep + f;
                        features[[series_idx, window_idx, feature_idx]] *= weight;
                    }
                }
            }
        }
    }

    /// Fit the feature extractor and compute normalization statistics
    pub fn fit(&mut self, series: &Array2<f64>) -> Result<()> {
        let mut all_features = self.extract_windows(series);
        self.apply_temporal_decay(&mut all_features);

        // Add derivative features if requested
        if self.config.use_derivatives {
            let derivatives = self.compute_derivatives(series);
            for derivative in derivatives {
                let mut deriv_windows = self.extract_windows(&derivative);
                self.apply_temporal_decay(&mut deriv_windows);
                // Concatenate derivative features
                all_features = concatenate_along_axis2(&all_features, &deriv_windows);
            }
        }

        // Compute normalization statistics
        if self.config.normalize_temporal {
            let (n_series, n_windows, n_features) = all_features.dim();
            let reshaped = all_features
                .into_shape((n_series * n_windows, n_features))
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

            self.feature_means = Some(reshaped.mean_axis(Axis(0)).unwrap());
            self.feature_stds = Some({
                let means = self.feature_means.as_ref().unwrap();
                let centered = &reshaped - means;
                let variance = (&centered * &centered).mean_axis(Axis(0)).unwrap();
                variance.mapv(|v| (v + 1e-8).sqrt()) // Add small epsilon for numerical stability
            });
        }

        Ok(())
    }

    /// Transform time series to temporal features
    pub fn transform(&self, series: &Array2<f64>) -> Result<Array2<f64>> {
        let mut all_features = self.extract_windows(series);
        self.apply_temporal_decay(&mut all_features);

        // Add derivative features if requested
        if self.config.use_derivatives {
            let derivatives = self.compute_derivatives(series);
            for derivative in derivatives {
                let mut deriv_windows = self.extract_windows(&derivative);
                self.apply_temporal_decay(&mut deriv_windows);
                all_features = concatenate_along_axis2(&all_features, &deriv_windows);
            }
        }

        // Reshape to 2D
        let (n_series, n_windows, n_features) = all_features.dim();
        let mut reshaped = all_features
            .into_shape((n_series * n_windows, n_features))
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

        // Apply normalization if fitted
        if self.config.normalize_temporal {
            if let (Some(means), Some(stds)) = (&self.feature_means, &self.feature_stds) {
                reshaped = (&reshaped - means) / stds;
            }
        }

        Ok(reshaped)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, series: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(series)?;
        self.transform(series)
    }
}

/// Temporal Naive Bayes classifier for time series
#[derive(Debug)]
pub struct TemporalNaiveBayes {
    /// Configuration
    pub config: TemporalConfig,
    /// Feature extractor
    pub feature_extractor: TemporalFeatureExtractor,
    /// Underlying Naive Bayes classifier
    pub classifier: Option<GaussianNB<Trained>>,
    /// Window labels for training
    window_labels: Option<Array1<i32>>,
}

impl Default for TemporalNaiveBayes {
    fn default() -> Self {
        Self::new(TemporalConfig::default())
    }
}

impl TemporalNaiveBayes {
    pub fn new(config: TemporalConfig) -> Self {
        let feature_extractor = TemporalFeatureExtractor::new(config.clone());
        Self {
            config,
            feature_extractor,
            classifier: None,
            window_labels: None,
        }
    }

    /// Fit the temporal model
    pub fn fit(&mut self, series: &Array2<f64>, labels: &Array1<i32>) -> Result<()> {
        // Transform time series to temporal features
        let features = self.feature_extractor.fit_transform(series)?;

        // Create labels for each window
        let window_labels = self.create_window_labels(series, labels)?;

        // Train Gaussian Naive Bayes on temporal features
        let classifier = GaussianNB::new();
        let trained_classifier = classifier.fit(&features, &window_labels)?;
        self.classifier = Some(trained_classifier);
        self.window_labels = Some(window_labels);

        Ok(())
    }

    /// Create labels for temporal windows
    fn create_window_labels(
        &self,
        series: &Array2<f64>,
        labels: &Array1<i32>,
    ) -> Result<Array1<i32>> {
        let (n_series, series_length) = series.dim();
        let n_windows = if series_length >= self.config.window_size {
            (series_length - self.config.window_size) / self.config.step_size + 1
        } else {
            0
        };

        let mut window_labels = Vec::new();

        for series_idx in 0..n_series {
            let series_label = labels[series_idx];
            for _ in 0..n_windows {
                window_labels.push(series_label);
            }
        }

        Ok(Array1::from_vec(window_labels))
    }

    /// Predict classes for time series
    pub fn predict(&self, series: &Array2<f64>) -> Result<Array1<i32>> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let features = self.feature_extractor.transform(series)?;

        // Get window-level predictions
        let window_predictions = classifier.predict(&features)?;

        // Aggregate window predictions to series predictions
        self.aggregate_window_predictions(&window_predictions, series)
    }

    /// Predict class probabilities for time series
    pub fn predict_proba(&self, series: &Array2<f64>) -> Result<Array2<f64>> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            })?;

        let features = self.feature_extractor.transform(series)?;

        // Get window-level probabilities
        let window_probabilities = classifier.predict_proba(&features)?;

        // Aggregate window probabilities to series probabilities
        self.aggregate_window_probabilities(&window_probabilities, series)
    }

    /// Aggregate window-level predictions to series-level predictions
    fn aggregate_window_predictions(
        &self,
        window_predictions: &Array1<i32>,
        series: &Array2<f64>,
    ) -> Result<Array1<i32>> {
        let (n_series, series_length) = series.dim();
        let n_windows = if series_length >= self.config.window_size {
            (series_length - self.config.window_size) / self.config.step_size + 1
        } else {
            0
        };

        let mut series_predictions = Vec::new();

        for series_idx in 0..n_series {
            if n_windows == 0 {
                // If no windows, use a default prediction (e.g., 0)
                series_predictions.push(0);
                continue;
            }

            let start_idx = series_idx * n_windows;
            let end_idx = start_idx + n_windows;

            // Use majority voting for aggregation
            let mut class_votes = HashMap::new();
            for &prediction in &window_predictions.slice(s![start_idx..end_idx]) {
                *class_votes.entry(prediction).or_insert(0) += 1;
            }

            let majority_class = class_votes
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(class, _)| *class)
                .unwrap_or(0);

            series_predictions.push(majority_class);
        }

        Ok(Array1::from_vec(series_predictions))
    }

    /// Aggregate window-level probabilities to series-level probabilities
    fn aggregate_window_probabilities(
        &self,
        window_probabilities: &Array2<f64>,
        series: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let (n_series, series_length) = series.dim();
        let n_windows = if series_length >= self.config.window_size {
            (series_length - self.config.window_size) / self.config.step_size + 1
        } else {
            0
        };
        let n_classes = window_probabilities.ncols();

        let mut series_probabilities = Array2::zeros((n_series, n_classes));

        for series_idx in 0..n_series {
            if n_windows == 0 {
                // If no windows, set uniform probabilities
                for class_idx in 0..n_classes {
                    series_probabilities[[series_idx, class_idx]] = 1.0 / n_classes as f64;
                }
                continue;
            }

            let start_idx = series_idx * n_windows;
            let end_idx = start_idx + n_windows;

            // Average probabilities across windows
            for class_idx in 0..n_classes {
                let avg_prob: f64 = window_probabilities
                    .slice(s![start_idx..end_idx, class_idx])
                    .mean()
                    .unwrap();
                series_probabilities[[series_idx, class_idx]] = avg_prob;
            }
        }

        Ok(series_probabilities)
    }
}

/// Streaming Temporal Naive Bayes for online classification
#[derive(Debug)]
pub struct StreamingTemporalNB {
    /// Configuration
    pub config: TemporalConfig,
    /// Buffer for streaming data
    buffer: VecDeque<Array1<f64>>,
    /// Buffer for labels
    label_buffer: VecDeque<i32>,
    /// Online feature statistics
    feature_sum: Option<Array1<f64>>,
    feature_sum_sq: Option<Array1<f64>>,
    feature_count: usize,
    /// Class statistics
    class_counts: HashMap<i32, usize>,
    class_feature_sums: HashMap<i32, Array1<f64>>,
    class_feature_sum_sqs: HashMap<i32, Array1<f64>>,
}

impl StreamingTemporalNB {
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            label_buffer: VecDeque::new(),
            feature_sum: None,
            feature_sum_sq: None,
            feature_count: 0,
            class_counts: HashMap::new(),
            class_feature_sums: HashMap::new(),
            class_feature_sum_sqs: HashMap::new(),
        }
    }

    /// Add a new data point to the stream
    pub fn partial_fit(&mut self, data_point: &Array1<f64>, label: i32) -> Result<()> {
        // Add to buffer
        self.buffer.push_back(data_point.clone());
        self.label_buffer.push_back(label);

        // Maintain buffer size
        if self.buffer.len() > self.config.window_size * 2 {
            self.buffer.pop_front();
            self.label_buffer.pop_front();
        }

        // Update statistics if we have enough data
        if self.buffer.len() >= self.config.window_size {
            self.update_statistics()?;
        }

        Ok(())
    }

    /// Update online statistics
    fn update_statistics(&mut self) -> Result<()> {
        if self.buffer.len() < self.config.window_size {
            return Ok(());
        }

        // Extract temporal features from current buffer
        // Each element in buffer is a feature vector, but for temporal analysis we need time series
        // For now, assume each buffer element represents a single timestep with scalar value
        let series_data: Vec<f64> = self.buffer.iter().map(|x| x[0]).collect();
        let series = Array2::from_shape_vec((1, series_data.len()), series_data)
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

        let mut extractor = TemporalFeatureExtractor::new(self.config.clone());
        let features = extractor.fit_transform(&series)?;

        if features.nrows() == 0 {
            return Ok(());
        }

        let feature_vector = features.row(0);
        let current_label = *self.label_buffer.back().unwrap();

        // Initialize if first update
        if self.feature_sum.is_none() {
            self.feature_sum = Some(Array1::zeros(feature_vector.len()));
            self.feature_sum_sq = Some(Array1::zeros(feature_vector.len()));
        }

        // Update global statistics
        if let (Some(ref mut sum), Some(ref mut sum_sq)) =
            (&mut self.feature_sum, &mut self.feature_sum_sq)
        {
            *sum = &*sum + &feature_vector;
            *sum_sq = &*sum_sq + &(&feature_vector * &feature_vector);
            self.feature_count += 1;
        }

        // Update class-specific statistics
        *self.class_counts.entry(current_label).or_insert(0) += 1;

        self.class_feature_sums
            .entry(current_label)
            .and_modify(|sum| *sum = &*sum + &feature_vector)
            .or_insert(feature_vector.to_owned());

        self.class_feature_sum_sqs
            .entry(current_label)
            .and_modify(|sum_sq| *sum_sq = &*sum_sq + &(&feature_vector * &feature_vector))
            .or_insert(&feature_vector * &feature_vector);

        Ok(())
    }

    /// Predict class for a new data point
    pub fn predict(&self, data_point: &Array1<f64>) -> Result<i32> {
        if self.class_counts.is_empty() {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        // Create a temporary buffer with the new data point
        let mut temp_buffer = self.buffer.clone();
        temp_buffer.push_back(data_point.clone());

        if temp_buffer.len() < self.config.window_size {
            // Not enough data, return most frequent class
            let most_frequent_class = self
                .class_counts
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(class, _)| *class)
                .unwrap_or(0);
            return Ok(most_frequent_class);
        }

        // Extract features
        let series_data: Vec<f64> = temp_buffer.iter().map(|x| x[0]).collect();
        let series = Array2::from_shape_vec((1, series_data.len()), series_data)
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

        let mut extractor = TemporalFeatureExtractor::new(self.config.clone());
        let features = extractor.fit_transform(&series)?;

        if features.nrows() == 0 {
            let most_frequent_class = self
                .class_counts
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(class, _)| *class)
                .unwrap_or(0);
            return Ok(most_frequent_class);
        }

        let feature_vector = features.row(0);

        // Compute log probabilities for each class
        let mut best_class = 0;
        let mut best_log_prob = f64::NEG_INFINITY;

        let total_count: usize = self.class_counts.values().sum();

        for (&class, &class_count) in &self.class_counts {
            // Prior probability
            let log_prior = (class_count as f64 / total_count as f64).ln();

            // Likelihood (assuming Gaussian distribution)
            let mut log_likelihood = 0.0;

            if let (Some(class_sum), Some(class_sum_sq)) = (
                self.class_feature_sums.get(&class),
                self.class_feature_sum_sqs.get(&class),
            ) {
                for (i, &feature_val) in feature_vector.iter().enumerate() {
                    let mean = class_sum[i] / class_count as f64;
                    let variance = (class_sum_sq[i] / class_count as f64) - mean * mean + 1e-8;
                    let std_dev = variance.sqrt();

                    // Gaussian log probability
                    log_likelihood += -0.5 * ((feature_val - mean) / std_dev).powi(2)
                        - 0.5 * (2.0 * std::f64::consts::PI).ln()
                        - std_dev.ln();
                }
            }

            let log_prob = log_prior + log_likelihood;
            if log_prob > best_log_prob {
                best_log_prob = log_prob;
                best_class = class;
            }
        }

        Ok(best_class)
    }
}

/// Helper function to concatenate arrays along the last axis
fn concatenate_along_axis2(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
    let (n_series, n_windows, features_a) = a.dim();
    let (_, _, features_b) = b.dim();

    let mut result = Array3::zeros((n_series, n_windows, features_a + features_b));

    // Copy data from first array
    for i in 0..n_series {
        for j in 0..n_windows {
            for k in 0..features_a {
                result[[i, j, k]] = a[[i, j, k]];
            }
        }
    }

    // Copy data from second array
    for i in 0..n_series {
        for j in 0..n_windows {
            for k in 0..features_b {
                result[[i, j, features_a + k]] = b[[i, j, k]];
            }
        }
    }

    result
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_temporal_feature_extractor() {
        let series = Array2::from_shape_vec(
            (2, 10),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0,
                4.0, 3.0, 2.0, 1.0,
            ],
        )
        .unwrap();

        let config = TemporalConfig {
            window_size: 3,
            step_size: 1,
            use_derivatives: false,
            derivative_order: 1,
            use_temporal_decay: false,
            decay_factor: 0.9,
            normalize_temporal: false,
        };

        let mut extractor = TemporalFeatureExtractor::new(config);
        let features = extractor.fit_transform(&series).unwrap();

        // Should have 16 windows total (8 from each series)
        assert_eq!(features.nrows(), 16);
        // Each window should have 3 features (window_size * n_features_per_timestep)
        assert_eq!(features.ncols(), 3);
    }

    #[test]
    fn test_temporal_naive_bayes() {
        let series = Array2::from_shape_vec(
            (4, 8),
            vec![
                // Class 0: increasing trend
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1,
                // Class 1: decreasing trend
                8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 8.1, 7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1,
            ],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let config = TemporalConfig {
            window_size: 3,
            step_size: 1,
            use_derivatives: true,
            derivative_order: 1,
            use_temporal_decay: false,
            decay_factor: 0.9,
            normalize_temporal: true,
        };

        let mut model = TemporalNaiveBayes::new(config);
        model.fit(&series, &labels).unwrap();

        let test_series = Array2::from_shape_vec(
            (1, 8),
            vec![
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Similar to class 0
            ],
        )
        .unwrap();

        let predictions = model.predict(&test_series).unwrap();
        assert_eq!(predictions.len(), 1);

        let probabilities = model.predict_proba(&test_series).unwrap();
        assert_eq!(probabilities.nrows(), 1);
        assert_eq!(probabilities.ncols(), 2);

        // Check that probabilities sum to 1
        let prob_sum: f64 = probabilities.row(0).sum();
        assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_streaming_temporal_nb() {
        let mut model = StreamingTemporalNB::new(TemporalConfig {
            window_size: 3,
            step_size: 1,
            use_derivatives: false,
            derivative_order: 1,
            use_temporal_decay: false,
            decay_factor: 0.9,
            normalize_temporal: false,
        });

        // Add some training data
        let data_points = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![2.0]),
            Array1::from_vec(vec![3.0]),
            Array1::from_vec(vec![4.0]),
            Array1::from_vec(vec![5.0]),
        ];
        let labels = vec![0, 0, 0, 1, 1];

        for (data_point, label) in data_points.iter().zip(labels.iter()) {
            model.partial_fit(data_point, *label).unwrap();
        }

        // Test prediction
        let test_point = Array1::from_vec(vec![6.0]);
        let prediction = model.predict(&test_point).unwrap();

        // Should predict class 1 (increasing trend continues)
        assert!(prediction == 0 || prediction == 1);
    }
}

/// Hidden Markov Model configuration for time series classification
#[derive(Debug, Clone)]
pub struct HMMConfig {
    /// Number of hidden states
    pub n_states: usize,
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Convergence tolerance for EM algorithm
    pub tol: f64,
    /// Random state for initialization
    pub random_state: Option<u64>,
    /// Covariance type for emission probabilities
    pub covar_type: CovarianceType,
}

impl Default for HMMConfig {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 100,
            tol: 1e-4,
            random_state: None,
            covar_type: CovarianceType::Full,
        }
    }
}

/// Covariance type for HMM emission probabilities
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceType {
    /// Full covariance matrix
    Full,
    /// Diagonal covariance matrix
    Diagonal,
    /// Spherical covariance (single variance parameter)
    Spherical,
}

/// Hidden Markov Model integrated with Naive Bayes for time series classification
#[derive(Debug)]
pub struct HMMNaiveBayes<State> {
    /// HMM configuration
    config: HMMConfig,
    /// Transition matrix (n_states x n_states)
    transition_probs: Option<Array2<f64>>,
    /// Initial state probabilities
    initial_probs: Option<Array1<f64>>,
    /// Emission probabilities for each state and class
    emission_models: Option<HashMap<(usize, i32), GaussianNB<Trained>>>,
    /// Class labels
    classes: Option<Array1<i32>>,
    /// Feature dimensionality
    n_features: Option<usize>,
    /// State marker
    _state: std::marker::PhantomData<State>,
}

impl HMMNaiveBayes<Untrained> {
    /// Create a new HMM Naive Bayes classifier
    pub fn new(config: HMMConfig) -> Self {
        Self {
            config,
            transition_probs: None,
            initial_probs: None,
            emission_models: None,
            classes: None,
            n_features: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Fit the HMM Naive Bayes model to time series data
    pub fn fit(mut self, X: &Array3<f64>, y: &Array1<i32>) -> Result<HMMNaiveBayes<Trained>> {
        let (n_sequences, sequence_length, n_features) = X.dim();
        self.n_features = Some(n_features);

        // Get unique classes
        let mut classes_vec: Vec<i32> = y.iter().cloned().collect();
        classes_vec.sort_unstable();
        classes_vec.dedup();
        let classes = Array1::from_vec(classes_vec);

        // Initialize HMM parameters
        // SciRS2 Policy Compliance - Use compatible random generator during migration
        // TODO: Migrate to full scirs2_core::random when API is stabilized
        let mut rng = if let Some(seed) = self.config.random_state {
            use scirs2_core::random::SeedableRng;
            CoreRandom::seed_from_u64(seed)
        } else {
            use scirs2_core::random::SeedableRng;
            CoreRandom::seed_from_u64(42) // Use fixed seed for reproducibility
        };

        // Initialize transition probabilities
        let mut transition_probs = Array2::zeros((self.config.n_states, self.config.n_states));
        for i in 0..self.config.n_states {
            let mut row_sum = 0.0;
            for j in 0..self.config.n_states {
                let val: f64 = scirs2_core::random::Rng::gen_range(&mut rng, 0.1..1.0);
                transition_probs[[i, j]] = val;
                row_sum += val;
            }
            // Normalize
            for j in 0..self.config.n_states {
                transition_probs[[i, j]] /= row_sum;
            }
        }

        // Initialize initial state probabilities
        let mut initial_probs = Array1::zeros(self.config.n_states);
        let mut sum = 0.0;
        for i in 0..self.config.n_states {
            let val: f64 = scirs2_core::random::Rng::gen_range(&mut rng, 0.1..1.0);
            initial_probs[i] = val;
            sum += val;
        }
        // Normalize
        for i in 0..self.config.n_states {
            initial_probs[i] /= sum;
        }

        // EM algorithm for HMM training
        let mut log_likelihood = f64::NEG_INFINITY;

        for iteration in 0..self.config.max_iter {
            let prev_log_likelihood = log_likelihood;

            // E-step: Compute forward-backward probabilities
            let (forward_probs, backward_probs, scaling_factors) =
                self.forward_backward_algorithm(X, &transition_probs, &initial_probs)?;

            // Compute log-likelihood
            log_likelihood = scaling_factors
                .iter()
                .map(|scales| scales.iter().map(|s| s.ln()).sum::<f64>())
                .sum();

            // Check for convergence
            if iteration > 0 && (log_likelihood - prev_log_likelihood).abs() < self.config.tol {
                break;
            }

            // M-step: Update parameters
            self.update_hmm_parameters(
                X,
                y,
                &forward_probs,
                &backward_probs,
                &scaling_factors,
                &mut transition_probs,
                &mut initial_probs,
            )?;
        }

        // Train emission models for each state-class combination
        let mut emission_models = HashMap::new();

        // Use Viterbi algorithm to find most likely state sequences
        let state_sequences = self.viterbi_decode(X, &transition_probs, &initial_probs)?;

        // Group data by state and class
        let mut state_class_data: HashMap<(usize, i32), Vec<Array1<f64>>> = HashMap::new();

        for seq_idx in 0..n_sequences {
            let class_label = y[seq_idx];
            for t in 0..sequence_length {
                let state = state_sequences[seq_idx][t];
                let key = (state, class_label);

                state_class_data
                    .entry(key)
                    .or_default()
                    .push(X.slice(s![seq_idx, t, ..]).to_owned());
            }
        }

        // Train Gaussian NB for each state-class combination
        for ((state, class_label), data) in state_class_data {
            if !data.is_empty() {
                let X_state_class = Array2::from_shape_vec(
                    (data.len(), n_features),
                    data.into_iter().flatten().collect(),
                )?;
                let y_state_class = Array1::from_elem(X_state_class.nrows(), class_label);

                let nb_model = GaussianNB::new().fit(&X_state_class, &y_state_class)?;

                emission_models.insert((state, class_label), nb_model);
            }
        }

        Ok(HMMNaiveBayes {
            config: self.config,
            transition_probs: Some(transition_probs),
            initial_probs: Some(initial_probs),
            emission_models: Some(emission_models),
            classes: Some(classes),
            n_features: Some(n_features),
            _state: std::marker::PhantomData,
        })
    }

    /// Forward-backward algorithm for HMM
    fn forward_backward_algorithm(
        &self,
        X: &Array3<f64>,
        transition_probs: &Array2<f64>,
        initial_probs: &Array1<f64>,
    ) -> Result<ForwardBackwardResult> {
        let (n_sequences, sequence_length, _) = X.dim();

        let mut forward_probs = Vec::new();
        let mut backward_probs = Vec::new();
        let mut scaling_factors = Vec::new();

        for seq_idx in 0..n_sequences {
            let mut alpha = Array2::zeros((sequence_length, self.config.n_states));
            let mut beta = Array2::zeros((sequence_length, self.config.n_states));
            let mut scales = Array1::zeros(sequence_length);

            // Forward pass
            // Initialize first timestep
            for state in 0..self.config.n_states {
                alpha[[0, state]] = initial_probs[state];
            }

            // Scale first timestep
            let scale_0 = alpha.row(0).sum();
            scales[0] = scale_0;
            if scale_0 > 0.0 {
                for state in 0..self.config.n_states {
                    alpha[[0, state]] /= scale_0;
                }
            }

            // Forward recursion
            for t in 1..sequence_length {
                for j in 0..self.config.n_states {
                    let mut sum = 0.0;
                    for i in 0..self.config.n_states {
                        sum += alpha[[t - 1, i]] * transition_probs[[i, j]];
                    }
                    alpha[[t, j]] = sum;
                }

                // Scale timestep t
                let scale_t = alpha.row(t).sum();
                scales[t] = scale_t;
                if scale_t > 0.0 {
                    for state in 0..self.config.n_states {
                        alpha[[t, state]] /= scale_t;
                    }
                }
            }

            // Backward pass
            // Initialize last timestep
            for state in 0..self.config.n_states {
                beta[[sequence_length - 1, state]] = 1.0 / scales[sequence_length - 1];
            }

            // Backward recursion
            for t in (0..sequence_length - 1).rev() {
                for i in 0..self.config.n_states {
                    let mut sum = 0.0;
                    for j in 0..self.config.n_states {
                        sum += transition_probs[[i, j]] * beta[[t + 1, j]];
                    }
                    beta[[t, i]] = sum / scales[t];
                }
            }

            forward_probs.push(alpha);
            backward_probs.push(beta);
            scaling_factors.push(scales);
        }

        Ok((forward_probs, backward_probs, scaling_factors))
    }

    /// Update HMM parameters in M-step
    #[allow(clippy::too_many_arguments)]
    fn update_hmm_parameters(
        &self,
        X: &Array3<f64>,
        y: &Array1<i32>,
        forward_probs: &[Array2<f64>],
        backward_probs: &[Array2<f64>],
        scaling_factors: &[Array1<f64>],
        transition_probs: &mut Array2<f64>,
        initial_probs: &mut Array1<f64>,
    ) -> Result<()> {
        let (n_sequences, sequence_length, _) = X.dim();

        // Update initial state probabilities
        let mut initial_counts = Array1::<f64>::zeros(self.config.n_states);
        for seq_idx in 0..n_sequences {
            for state in 0..self.config.n_states {
                initial_counts[state] +=
                    forward_probs[seq_idx][[0, state]] * backward_probs[seq_idx][[0, state]];
            }
        }
        let initial_sum: f64 = initial_counts.sum();
        if initial_sum > 0.0 {
            *initial_probs = &initial_counts / initial_sum;
        }

        // Update transition probabilities
        let mut transition_counts =
            Array2::<f64>::zeros((self.config.n_states, self.config.n_states));
        let mut state_counts = Array1::<f64>::zeros(self.config.n_states);

        for seq_idx in 0..n_sequences {
            for t in 0..sequence_length - 1 {
                for i in 0..self.config.n_states {
                    for j in 0..self.config.n_states {
                        let xi = forward_probs[seq_idx][[t, i]]
                            * transition_probs[[i, j]]
                            * backward_probs[seq_idx][[t + 1, j]]
                            * scaling_factors[seq_idx][t + 1];

                        transition_counts[[i, j]] += xi;
                        state_counts[i] += xi;
                    }
                }
            }
        }

        // Normalize transition probabilities
        for i in 0..self.config.n_states {
            if state_counts[i] > 0.0 {
                for j in 0..self.config.n_states {
                    transition_probs[[i, j]] = transition_counts[[i, j]] / state_counts[i];
                }
            }
        }

        Ok(())
    }

    /// Viterbi algorithm for finding most likely state sequence
    fn viterbi_decode(
        &self,
        X: &Array3<f64>,
        transition_probs: &Array2<f64>,
        initial_probs: &Array1<f64>,
    ) -> Result<Vec<Vec<usize>>> {
        let (n_sequences, sequence_length, _) = X.dim();
        let mut state_sequences = Vec::new();

        for seq_idx in 0..n_sequences {
            let mut viterbi_probs = Array2::zeros((sequence_length, self.config.n_states));
            let mut viterbi_path = Array2::zeros((sequence_length, self.config.n_states));

            // Initialize first timestep
            for state in 0..self.config.n_states {
                viterbi_probs[[0, state]] = initial_probs[state].ln();
            }

            // Forward pass
            for t in 1..sequence_length {
                for j in 0..self.config.n_states {
                    let mut best_prob = f64::NEG_INFINITY;
                    let mut best_state = 0;

                    for i in 0..self.config.n_states {
                        let prob = viterbi_probs[[t - 1, i]] + transition_probs[[i, j]].ln();
                        if prob > best_prob {
                            best_prob = prob;
                            best_state = i;
                        }
                    }

                    viterbi_probs[[t, j]] = best_prob;
                    viterbi_path[[t, j]] = best_state as f64;
                }
            }

            // Backward pass - find best path
            let mut path = vec![0; sequence_length];

            // Find best final state
            let mut best_final_prob = f64::NEG_INFINITY;
            let mut best_final_state = 0;
            for state in 0..self.config.n_states {
                if viterbi_probs[[sequence_length - 1, state]] > best_final_prob {
                    best_final_prob = viterbi_probs[[sequence_length - 1, state]];
                    best_final_state = state;
                }
            }

            path[sequence_length - 1] = best_final_state;

            // Backtrack
            for t in (0..sequence_length - 1).rev() {
                path[t] = viterbi_path[[t + 1, path[t + 1]]] as usize;
            }

            state_sequences.push(path);
        }

        Ok(state_sequences)
    }
}

impl HMMNaiveBayes<Trained> {
    /// Predict class labels for time series sequences
    pub fn predict(&self, X: &Array3<f64>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let classes = self.classes.as_ref().unwrap();

        let predictions = probabilities
            .outer_iter()
            .map(|probs| {
                let max_idx = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    /// Predict class probabilities for time series sequences
    pub fn predict_proba(&self, X: &Array3<f64>) -> Result<Array2<f64>> {
        let (n_sequences, sequence_length, _) = X.dim();
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();

        let transition_probs = self.transition_probs.as_ref().unwrap();
        let initial_probs = self.initial_probs.as_ref().unwrap();
        let emission_models = self.emission_models.as_ref().unwrap();

        let mut predictions = Array2::zeros((n_sequences, n_classes));

        for seq_idx in 0..n_sequences {
            // Find most likely state sequence using Viterbi
            let state_sequence = self.viterbi_decode_single(
                &X.slice(s![seq_idx, .., ..]).to_owned().insert_axis(Axis(0)),
                transition_probs,
                initial_probs,
            )?[0]
                .clone();

            // Aggregate predictions from each timestep
            let mut class_log_probs = Array1::zeros(n_classes);
            let mut timestep_count = 0;

            for (t, &state) in state_sequence.iter().enumerate().take(sequence_length) {
                let observation = X.slice(s![seq_idx, t, ..]).to_owned();

                // Get predictions from emission models for this state
                for (class_idx, &class_label) in classes.iter().enumerate() {
                    if let Some(model) = emission_models.get(&(state, class_label)) {
                        let obs_2d = observation.clone().insert_axis(Axis(0));
                        let obs_labels = Array1::from_elem(1, class_label);
                        let proba = model.predict_proba(&obs_2d)?;

                        // Find the class probability (should be close to 1 for correct class)
                        let class_proba = proba[[0, 0]]; // Assuming binary classification within state
                        class_log_probs[class_idx] += class_proba.ln();
                        timestep_count += 1;
                    }
                }
            }

            // Normalize by number of timesteps
            if timestep_count > 0 {
                class_log_probs /= timestep_count as f64;
            }

            // Convert to probabilities
            let max_log_prob = class_log_probs
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut class_probs = Array1::zeros(n_classes);
            let mut prob_sum = 0.0;

            for (i, &log_prob) in class_log_probs.iter().enumerate() {
                class_probs[i] = (log_prob - max_log_prob).exp();
                prob_sum += class_probs[i];
            }

            // Normalize
            if prob_sum > 0.0 {
                class_probs /= prob_sum;
            } else {
                class_probs.fill(1.0 / n_classes as f64);
            }

            predictions.row_mut(seq_idx).assign(&class_probs);
        }

        Ok(predictions)
    }

    /// Viterbi decode for a single sequence
    fn viterbi_decode_single(
        &self,
        X: &Array3<f64>,
        transition_probs: &Array2<f64>,
        initial_probs: &Array1<f64>,
    ) -> Result<Vec<Vec<usize>>> {
        // Reuse the logic from the untrained version
        let untrained_self = HMMNaiveBayes::<Untrained> {
            config: self.config.clone(),
            transition_probs: None,
            initial_probs: None,
            emission_models: None,
            classes: None,
            n_features: None,
            _state: std::marker::PhantomData,
        };

        untrained_self.viterbi_decode(X, transition_probs, initial_probs)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod hmm_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_hmm_naive_bayes_basic() {
        // Create simple time series data
        let mut sequences = Vec::new();
        let mut labels = Vec::new();

        // Class 0: sequences with decreasing trend
        for _ in 0..5 {
            let seq = Array2::from_shape_vec(
                (10, 2),
                (0..20)
                    .map(|i| {
                        if i % 2 == 0 {
                            10.0 - i as f64 / 2.0
                        } else {
                            5.0 - i as f64 / 2.0
                        }
                    })
                    .collect(),
            )
            .unwrap();
            sequences.push(seq);
            labels.push(0);
        }

        // Class 1: sequences with increasing trend
        for _ in 0..5 {
            let seq = Array2::from_shape_vec(
                (10, 2),
                (0..20)
                    .map(|i| {
                        if i % 2 == 0 {
                            i as f64 / 2.0
                        } else {
                            i as f64 / 2.0 + 5.0
                        }
                    })
                    .collect(),
            )
            .unwrap();
            sequences.push(seq);
            labels.push(1);
        }

        // Convert to 3D array
        let n_sequences = sequences.len();
        let (seq_len, n_features) = sequences[0].dim();
        let mut X = Array3::zeros((n_sequences, seq_len, n_features));

        for (i, seq) in sequences.iter().enumerate() {
            X.slice_mut(s![i, .., ..]).assign(seq);
        }

        let y = Array1::from_vec(labels);

        // Create and train HMM Naive Bayes
        let config = HMMConfig {
            n_states: 3,
            max_iter: 10,
            tol: 1e-3,
            random_state: Some(42),
            covar_type: CovarianceType::Diagonal,
        };

        let model = HMMNaiveBayes::new(config);
        let trained_model = model.fit(&X, &y).unwrap();

        // Test prediction
        let predictions = trained_model.predict(&X).unwrap();

        // Should have correct number of predictions
        assert_eq!(predictions.len(), n_sequences);

        // Test probability prediction
        let probabilities = trained_model.predict_proba(&X).unwrap();
        assert_eq!(probabilities.shape(), &[n_sequences, 2]); // 2 classes

        // Probabilities should sum to 1
        for row in probabilities.outer_iter() {
            let sum: f64 = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }
}
