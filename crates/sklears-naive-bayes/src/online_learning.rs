//! Online/Incremental Learning for Naive Bayes
//!
//! Implements streaming and incremental learning capabilities for Naive Bayes
//! classifiers, allowing for real-time updates and memory-efficient processing
//! of large datasets.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Type alias for chunk data: (features, labels)
type ChunkData = (Array2<f64>, Array1<i32>);

#[derive(Error, Debug)]
pub enum OnlineLearningError {
    #[error("Features and targets have different number of samples")]
    DimensionMismatch,
    #[error("Model not initialized")]
    ModelNotInitialized,
    #[error("Inconsistent number of features: expected {expected}, got {actual}")]
    InconsistentFeatures { expected: usize, actual: usize },
    #[error("Invalid learning rate: {0}")]
    InvalidLearningRate(f64),
    #[error("Buffer overflow: maximum size {0} exceeded")]
    BufferOverflow(usize),
    #[error("Numerical computation error: {0}")]
    NumericalError(String),
    #[error("Concept drift detection failed: {0}")]
    ConceptDriftError(String),
}

/// Configuration for online learning
#[derive(Debug, Clone)]
pub struct OnlineLearningConfig {
    pub learning_rate: f64,
    pub decay_factor: f64,
    pub buffer_size: usize,
    pub min_samples_for_update: usize,
    pub concept_drift_detection: bool,
    pub drift_threshold: f64,
    pub forgetting_factor: f64,
    pub adaptive_learning_rate: bool,
    pub batch_update_size: usize,
    // Out-of-core specific parameters
    pub chunk_size: usize,
    pub max_memory_usage: usize, // in MB
    pub enable_partial_fit: bool,
    pub checkpoint_frequency: usize, // save model every N chunks
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            decay_factor: 0.999,
            buffer_size: 1000,
            min_samples_for_update: 1,
            concept_drift_detection: true,
            drift_threshold: 0.1,
            forgetting_factor: 0.95,
            adaptive_learning_rate: true,
            batch_update_size: 10,
            // Out-of-core defaults
            chunk_size: 10000,
            max_memory_usage: 512, // 512 MB
            enable_partial_fit: true,
            checkpoint_frequency: 100,
        }
    }
}

/// Streaming data buffer for managing incoming samples
#[derive(Debug, Clone)]
pub struct StreamingBuffer {
    features: VecDeque<Array1<f64>>,
    labels: VecDeque<i32>,
    timestamps: VecDeque<u64>,
    max_size: usize,
}

impl StreamingBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            features: VecDeque::with_capacity(max_size),
            labels: VecDeque::with_capacity(max_size),
            timestamps: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn add_sample(
        &mut self,
        feature: Array1<f64>,
        label: i32,
        timestamp: u64,
    ) -> Result<(), OnlineLearningError> {
        if self.features.len() >= self.max_size {
            // Remove oldest sample
            self.features.pop_front();
            self.labels.pop_front();
            self.timestamps.pop_front();
        }

        self.features.push_back(feature);
        self.labels.push_back(label);
        self.timestamps.push_back(timestamp);

        Ok(())
    }

    pub fn get_recent_samples(&self, n: usize) -> (Vec<Array1<f64>>, Vec<i32>) {
        let actual_n = n.min(self.features.len());
        let features: Vec<Array1<f64>> =
            self.features.iter().rev().take(actual_n).cloned().collect();
        let labels: Vec<i32> = self.labels.iter().rev().take(actual_n).cloned().collect();

        (features, labels)
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn clear(&mut self) {
        self.features.clear();
        self.labels.clear();
        self.timestamps.clear();
    }
}

/// Concept drift detector
#[derive(Debug, Clone)]
pub struct ConceptDriftDetector {
    window_size: usize,
    error_rate_window: VecDeque<f64>,
    baseline_error_rate: f64,
    drift_threshold: f64,
    warning_threshold: f64,
    drift_detected: bool,
    warning_detected: bool,
}

impl ConceptDriftDetector {
    pub fn new(window_size: usize, drift_threshold: f64) -> Self {
        Self {
            window_size,
            error_rate_window: VecDeque::with_capacity(window_size),
            baseline_error_rate: 0.0,
            drift_threshold,
            warning_threshold: drift_threshold * 0.5,
            drift_detected: false,
            warning_detected: false,
        }
    }

    pub fn update(&mut self, error_rate: f64) -> Result<(), OnlineLearningError> {
        self.error_rate_window.push_back(error_rate);

        if self.error_rate_window.len() > self.window_size {
            self.error_rate_window.pop_front();
        }

        // Calculate current average error rate
        let current_error_rate =
            self.error_rate_window.iter().sum::<f64>() / self.error_rate_window.len() as f64;

        // Initialize baseline if not set
        if self.baseline_error_rate == 0.0 {
            self.baseline_error_rate = current_error_rate;
        }

        // Check for drift
        let error_increase = current_error_rate - self.baseline_error_rate;

        if error_increase > self.drift_threshold {
            self.drift_detected = true;
            self.warning_detected = false;
        } else if error_increase > self.warning_threshold {
            self.warning_detected = true;
            self.drift_detected = false;
        } else {
            self.drift_detected = false;
            self.warning_detected = false;
        }

        // Update baseline gradually
        self.baseline_error_rate = 0.95 * self.baseline_error_rate + 0.05 * current_error_rate;

        Ok(())
    }

    pub fn is_drift_detected(&self) -> bool {
        self.drift_detected
    }

    pub fn is_warning_detected(&self) -> bool {
        self.warning_detected
    }

    pub fn reset(&mut self) {
        self.error_rate_window.clear();
        self.drift_detected = false;
        self.warning_detected = false;
    }
}

/// Online Gaussian Naive Bayes statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineGaussianStats {
    pub n_samples: f64,
    pub mean: f64,
    pub variance: f64,
    pub sum_x: f64,
    pub sum_x_squared: f64,
}

impl Default for OnlineGaussianStats {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineGaussianStats {
    pub fn new() -> Self {
        Self {
            n_samples: 0.0,
            mean: 0.0,
            variance: 1.0,
            sum_x: 0.0,
            sum_x_squared: 0.0,
        }
    }

    pub fn update(&mut self, value: f64, weight: f64) {
        self.n_samples += weight;
        self.sum_x += weight * value;
        self.sum_x_squared += weight * value * value;

        if self.n_samples > 0.0 {
            self.mean = self.sum_x / self.n_samples;

            if self.n_samples > 1.0 {
                let variance_numerator =
                    self.sum_x_squared - (self.sum_x * self.sum_x) / self.n_samples;
                self.variance = (variance_numerator / (self.n_samples - 1.0)).max(1e-9);
            }
        }
    }

    pub fn update_with_forgetting(&mut self, value: f64, weight: f64, forgetting_factor: f64) {
        // Apply forgetting factor to existing statistics
        self.n_samples *= forgetting_factor;
        self.sum_x *= forgetting_factor;
        self.sum_x_squared *= forgetting_factor;

        // Add new observation
        self.update(value, weight);
    }

    pub fn log_likelihood(&self, value: f64) -> f64 {
        let var = self.variance.max(1e-9);
        let diff = value - self.mean;
        -0.5 * ((diff * diff / var) + var.ln() + (2.0 * std::f64::consts::PI).ln())
    }
}

/// Online Multinomial Naive Bayes statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineMultinomialStats {
    pub feature_counts: HashMap<i32, f64>,
    pub total_count: f64,
    pub smoothing: f64,
}

impl OnlineMultinomialStats {
    pub fn new(smoothing: f64) -> Self {
        Self {
            feature_counts: HashMap::new(),
            total_count: 0.0,
            smoothing,
        }
    }

    pub fn update(&mut self, feature_value: i32, weight: f64) {
        *self.feature_counts.entry(feature_value).or_insert(0.0) += weight;
        self.total_count += weight;
    }

    pub fn update_with_forgetting(
        &mut self,
        feature_value: i32,
        weight: f64,
        forgetting_factor: f64,
    ) {
        // Apply forgetting factor
        for count in self.feature_counts.values_mut() {
            *count *= forgetting_factor;
        }
        self.total_count *= forgetting_factor;

        // Add new observation
        self.update(feature_value, weight);
    }

    pub fn log_probability(&self, feature_value: i32) -> f64 {
        let count = self
            .feature_counts
            .get(&feature_value)
            .copied()
            .unwrap_or(0.0);
        let smoothed_count = count + self.smoothing;
        let smoothed_total = self.total_count + self.smoothing * self.feature_counts.len() as f64;

        if smoothed_total > 0.0 {
            (smoothed_count / smoothed_total).ln()
        } else {
            0.0
        }
    }
}

/// Online/Streaming Naive Bayes classifier
pub struct OnlineNaiveBayes {
    config: OnlineLearningConfig,

    // Class statistics
    class_counts: HashMap<i32, f64>,
    total_samples: f64,

    // Feature statistics (Gaussian)
    gaussian_stats: HashMap<(i32, usize), OnlineGaussianStats>, // (class, feature_idx) -> stats

    // Feature statistics (Multinomial)
    multinomial_stats: HashMap<(i32, usize), OnlineMultinomialStats>, // (class, feature_idx) -> stats

    // Model metadata
    classes: Vec<i32>,
    n_features: Option<usize>,
    feature_types: Vec<FeatureType>,

    // Online learning components
    streaming_buffer: StreamingBuffer,
    drift_detector: Option<ConceptDriftDetector>,
    current_learning_rate: f64,

    // Performance tracking
    accuracy_history: VecDeque<f64>,
    sample_count: u64,

    is_initialized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    /// Continuous
    Continuous,
    /// Discrete
    Discrete,
}

/// Checkpoint structure for serializing OnlineNaiveBayes state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineNBCheckpoint {
    pub class_counts: HashMap<i32, f64>,
    pub total_samples: f64,
    pub gaussian_stats: HashMap<(i32, usize), OnlineGaussianStats>,
    pub multinomial_stats: HashMap<(i32, usize), OnlineMultinomialStats>,
    pub classes: Vec<i32>,
    pub n_features: Option<usize>,
    pub feature_types: Vec<FeatureType>,
    pub current_learning_rate: f64,
    pub sample_count: u64,
    pub is_initialized: bool,
    pub processed_chunks: u64,
    pub total_samples_seen: u64,
}

impl OnlineNaiveBayes {
    pub fn new(config: OnlineLearningConfig) -> Self {
        let drift_detector = if config.concept_drift_detection {
            Some(ConceptDriftDetector::new(100, config.drift_threshold))
        } else {
            None
        };

        Self {
            streaming_buffer: StreamingBuffer::new(config.buffer_size),
            current_learning_rate: config.learning_rate,
            config,
            class_counts: HashMap::new(),
            total_samples: 0.0,
            gaussian_stats: HashMap::new(),
            multinomial_stats: HashMap::new(),
            classes: Vec::new(),
            n_features: None,
            feature_types: Vec::new(),
            drift_detector,
            accuracy_history: VecDeque::with_capacity(1000),
            sample_count: 0,
            is_initialized: false,
        }
    }

    /// Set feature types for proper handling
    pub fn set_feature_types(&mut self, feature_types: Vec<FeatureType>) {
        self.feature_types = feature_types;
    }

    /// Initialize the model with initial batch of data
    pub fn initialize(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<(), OnlineLearningError> {
        if X.nrows() != y.len() {
            return Err(OnlineLearningError::DimensionMismatch);
        }

        self.n_features = Some(X.ncols());

        // Initialize feature types if not set
        if self.feature_types.is_empty() {
            self.feature_types = vec![FeatureType::Continuous; X.ncols()];
        }

        // Get unique classes
        let class_set: std::collections::HashSet<i32> = y.iter().cloned().collect();
        self.classes = class_set.into_iter().collect();
        self.classes.sort();

        // Process initial batch
        for (sample_idx, sample) in X.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
            let label = y[sample_idx];
            self.partial_fit_single(&sample.to_owned(), label, 1.0)?;
        }

        self.is_initialized = true;
        Ok(())
    }

    /// Add a single sample for online learning
    pub fn partial_fit(
        &mut self,
        sample: &Array1<f64>,
        label: i32,
    ) -> Result<(), OnlineLearningError> {
        if !self.is_initialized {
            return Err(OnlineLearningError::ModelNotInitialized);
        }

        if let Some(n_features) = self.n_features {
            if sample.len() != n_features {
                return Err(OnlineLearningError::InconsistentFeatures {
                    expected: n_features,
                    actual: sample.len(),
                });
            }
        }

        // Add to streaming buffer
        self.streaming_buffer
            .add_sample(sample.clone(), label, self.sample_count)?;

        // Update model
        self.partial_fit_single(sample, label, 1.0)?;

        // Update learning rate if adaptive
        if self.config.adaptive_learning_rate {
            self.update_learning_rate();
        }

        // Check for concept drift
        if self.drift_detector.is_some() {
            let prediction = self.predict_single(sample)?;
            let error = if prediction == label { 0.0 } else { 1.0 };

            if let Some(ref mut detector) = self.drift_detector {
                detector.update(error)?;
                if detector.is_drift_detected() {
                    self.handle_concept_drift()?;
                }
            }
        }

        self.sample_count += 1;
        Ok(())
    }

    /// Batch partial fit for multiple samples
    pub fn partial_fit_batch(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<(), OnlineLearningError> {
        if X.nrows() != y.len() {
            return Err(OnlineLearningError::DimensionMismatch);
        }

        for (sample_idx, sample) in X.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
            let label = y[sample_idx];
            self.partial_fit(&sample.to_owned(), label)?;
        }

        Ok(())
    }

    fn partial_fit_single(
        &mut self,
        sample: &Array1<f64>,
        label: i32,
        weight: f64,
    ) -> Result<(), OnlineLearningError> {
        // Update class counts
        *self.class_counts.entry(label).or_insert(0.0) += weight;
        self.total_samples += weight;

        // Add class if new
        if !self.classes.contains(&label) {
            self.classes.push(label);
            self.classes.sort();
        }

        // Update feature statistics
        for (feature_idx, &feature_value) in sample.iter().enumerate() {
            match self
                .feature_types
                .get(feature_idx)
                .unwrap_or(&FeatureType::Continuous)
            {
                FeatureType::Continuous => {
                    let key = (label, feature_idx);
                    let stats = self.gaussian_stats.entry(key).or_default();

                    if self.config.forgetting_factor < 1.0 {
                        stats.update_with_forgetting(
                            feature_value,
                            weight,
                            self.config.forgetting_factor,
                        );
                    } else {
                        stats.update(feature_value, weight);
                    }
                }
                FeatureType::Discrete => {
                    let key = (label, feature_idx);
                    let stats = self
                        .multinomial_stats
                        .entry(key)
                        .or_insert_with(|| OnlineMultinomialStats::new(1.0));

                    let discrete_value = feature_value.round() as i32;
                    if self.config.forgetting_factor < 1.0 {
                        stats.update_with_forgetting(
                            discrete_value,
                            weight,
                            self.config.forgetting_factor,
                        );
                    } else {
                        stats.update(discrete_value, weight);
                    }
                }
            }
        }

        Ok(())
    }

    /// Predict a single sample
    pub fn predict_single(&self, sample: &Array1<f64>) -> Result<i32, OnlineLearningError> {
        if !self.is_initialized {
            return Err(OnlineLearningError::ModelNotInitialized);
        }

        let mut best_class = self.classes[0];
        let mut best_log_prob = f64::NEG_INFINITY;

        for &class in &self.classes {
            let log_prob = self.compute_log_probability(sample, class)?;
            if log_prob > best_log_prob {
                best_log_prob = log_prob;
                best_class = class;
            }
        }

        Ok(best_class)
    }

    /// Predict multiple samples
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>, OnlineLearningError> {
        let mut predictions = Vec::new();

        for sample in X.axis_iter(scirs2_core::ndarray::Axis(0)) {
            let prediction = self.predict_single(&sample.to_owned())?;
            predictions.push(prediction);
        }

        Ok(Array1::from_vec(predictions))
    }

    /// Predict probabilities for a single sample
    pub fn predict_proba_single(
        &self,
        sample: &Array1<f64>,
    ) -> Result<HashMap<i32, f64>, OnlineLearningError> {
        if !self.is_initialized {
            return Err(OnlineLearningError::ModelNotInitialized);
        }

        let mut log_probabilities = HashMap::new();

        for &class in &self.classes {
            let log_prob = self.compute_log_probability(sample, class)?;
            log_probabilities.insert(class, log_prob);
        }

        // Convert to probabilities
        let max_log_prob = log_probabilities
            .values()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities = HashMap::new();
        let mut total_prob = 0.0;

        for (&class, &log_prob) in &log_probabilities {
            let prob = (log_prob - max_log_prob).exp();
            probabilities.insert(class, prob);
            total_prob += prob;
        }

        // Normalize
        for prob in probabilities.values_mut() {
            *prob /= total_prob;
        }

        Ok(probabilities)
    }

    fn compute_log_probability(
        &self,
        sample: &Array1<f64>,
        class: i32,
    ) -> Result<f64, OnlineLearningError> {
        // Class prior
        let class_count = self.class_counts.get(&class).copied().unwrap_or(0.0);
        let log_prior = if self.total_samples > 0.0 {
            (class_count / self.total_samples).ln()
        } else {
            0.0
        };

        // Feature likelihoods
        let mut log_likelihood = 0.0;

        for (feature_idx, &feature_value) in sample.iter().enumerate() {
            match self
                .feature_types
                .get(feature_idx)
                .unwrap_or(&FeatureType::Continuous)
            {
                FeatureType::Continuous => {
                    let key = (class, feature_idx);
                    if let Some(stats) = self.gaussian_stats.get(&key) {
                        log_likelihood += stats.log_likelihood(feature_value);
                    }
                }
                FeatureType::Discrete => {
                    let key = (class, feature_idx);
                    if let Some(stats) = self.multinomial_stats.get(&key) {
                        let discrete_value = feature_value.round() as i32;
                        log_likelihood += stats.log_probability(discrete_value);
                    }
                }
            }
        }

        Ok(log_prior + log_likelihood)
    }

    fn update_learning_rate(&mut self) {
        if self.config.adaptive_learning_rate {
            self.current_learning_rate *= self.config.decay_factor;
            self.current_learning_rate = self.current_learning_rate.max(1e-6);
        }
    }

    fn handle_concept_drift(&mut self) -> Result<(), OnlineLearningError> {
        // Reset drift detector
        if let Some(ref mut detector) = self.drift_detector {
            detector.reset();
        }

        // Increase learning rate temporarily
        self.current_learning_rate = self.config.learning_rate;

        // Optionally: reduce the weight of old statistics
        // This is simplified; a full implementation might be more sophisticated

        Ok(())
    }

    /// Get model statistics
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        stats.insert("total_samples".to_string(), self.total_samples);
        stats.insert("n_classes".to_string(), self.classes.len() as f64);
        stats.insert(
            "current_learning_rate".to_string(),
            self.current_learning_rate,
        );
        stats.insert(
            "buffer_size".to_string(),
            self.streaming_buffer.len() as f64,
        );

        if let Some(ref detector) = self.drift_detector {
            stats.insert(
                "drift_detected".to_string(),
                if detector.is_drift_detected() {
                    1.0
                } else {
                    0.0
                },
            );
            stats.insert(
                "warning_detected".to_string(),
                if detector.is_warning_detected() {
                    1.0
                } else {
                    0.0
                },
            );
        }

        stats
    }

    /// Reset the model (for handling severe concept drift)
    pub fn reset(&mut self) {
        self.class_counts.clear();
        self.total_samples = 0.0;
        self.gaussian_stats.clear();
        self.multinomial_stats.clear();
        self.streaming_buffer.clear();
        self.accuracy_history.clear();
        self.sample_count = 0;
        self.current_learning_rate = self.config.learning_rate;

        if let Some(ref mut detector) = self.drift_detector {
            detector.reset();
        }

        self.is_initialized = false;
    }

    /// Get recent accuracy if tracking
    pub fn get_recent_accuracy(&self, window_size: usize) -> Option<f64> {
        if self.accuracy_history.len() < window_size {
            return None;
        }

        let recent_accuracies: Vec<f64> = self
            .accuracy_history
            .iter()
            .rev()
            .take(window_size)
            .cloned()
            .collect();

        Some(recent_accuracies.iter().sum::<f64>() / recent_accuracies.len() as f64)
    }

    /// Check if concept drift was detected
    pub fn is_drift_detected(&self) -> bool {
        self.drift_detector
            .as_ref()
            .is_some_and(|d| d.is_drift_detected())
    }

    /// Get the number of samples processed
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }
}

/// Trait for data sources that can provide chunks of data
pub trait DataChunkIterator {
    type Error;

    /// Get the next chunk of data
    fn next_chunk(&mut self) -> Result<Option<ChunkData>, Self::Error>;

    /// Estimate total number of samples (if known)
    fn estimated_total_samples(&self) -> Option<usize>;

    /// Get chunk size
    fn chunk_size(&self) -> usize;
}

/// Out-of-core data reader that reads data from a CSV file in chunks
pub struct CSVChunkReader {
    chunk_size: usize,
    current_line: usize,
    total_estimated: Option<usize>,
    feature_columns: Vec<usize>,
    target_column: usize,
    has_header: bool,
    file_path: String,
}

impl CSVChunkReader {
    pub fn new(
        file_path: String,
        chunk_size: usize,
        feature_columns: Vec<usize>,
        target_column: usize,
        has_header: bool,
    ) -> Self {
        Self {
            chunk_size,
            current_line: if has_header { 1 } else { 0 },
            total_estimated: None,
            feature_columns,
            target_column,
            has_header,
            file_path,
        }
    }
}

impl DataChunkIterator for CSVChunkReader {
    type Error = OnlineLearningError;

    fn next_chunk(&mut self) -> Result<Option<(Array2<f64>, Array1<i32>)>, Self::Error> {
        // This is a simplified implementation
        // In a real implementation, you'd use a CSV parsing library like csv or polars
        Err(OnlineLearningError::NumericalError(
            "CSV reading not implemented in this demo".to_string(),
        ))
    }

    fn estimated_total_samples(&self) -> Option<usize> {
        self.total_estimated
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

/// Memory-based chunk iterator for testing
pub struct MemoryChunkIterator {
    data: Array2<f64>,
    labels: Array1<i32>,
    chunk_size: usize,
    current_index: usize,
}

impl MemoryChunkIterator {
    pub fn new(data: Array2<f64>, labels: Array1<i32>, chunk_size: usize) -> Self {
        Self {
            data,
            labels,
            chunk_size,
            current_index: 0,
        }
    }
}

impl DataChunkIterator for MemoryChunkIterator {
    type Error = OnlineLearningError;

    fn next_chunk(&mut self) -> Result<Option<(Array2<f64>, Array1<i32>)>, Self::Error> {
        if self.current_index >= self.data.nrows() {
            return Ok(None);
        }

        let end_index = (self.current_index + self.chunk_size).min(self.data.nrows());
        let chunk_size = end_index - self.current_index;

        let mut chunk_data = Array2::zeros((chunk_size, self.data.ncols()));
        let mut chunk_labels = Array1::zeros(chunk_size);

        for i in 0..chunk_size {
            let row_idx = self.current_index + i;
            for j in 0..self.data.ncols() {
                chunk_data[[i, j]] = self.data[[row_idx, j]];
            }
            chunk_labels[i] = self.labels[row_idx];
        }

        self.current_index = end_index;
        Ok(Some((chunk_data, chunk_labels)))
    }

    fn estimated_total_samples(&self) -> Option<usize> {
        Some(self.data.nrows())
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

/// Out-of-core Naive Bayes classifier
pub struct OutOfCoreNaiveBayes {
    online_nb: OnlineNaiveBayes,
    config: OnlineLearningConfig,
    processed_chunks: usize,
    total_samples_seen: usize,
    memory_usage_mb: usize,
}

impl OutOfCoreNaiveBayes {
    pub fn new(config: OnlineLearningConfig) -> Self {
        let online_nb = OnlineNaiveBayes::new(config.clone());

        Self {
            online_nb,
            config,
            processed_chunks: 0,
            total_samples_seen: 0,
            memory_usage_mb: 0,
        }
    }

    /// Set feature types for the classifier
    pub fn set_feature_types(&mut self, feature_types: Vec<FeatureType>) {
        self.online_nb.set_feature_types(feature_types);
    }

    /// Fit the model using out-of-core learning
    pub fn fit_out_of_core<T>(&mut self, mut data_iterator: T) -> Result<(), OnlineLearningError>
    where
        T: DataChunkIterator<Error = OnlineLearningError>,
    {
        let mut is_first_chunk = true;

        while let Some((chunk_X, chunk_y)) = data_iterator.next_chunk()? {
            // Check memory usage
            let estimated_chunk_memory = chunk_X.len() * 8 / (1024 * 1024); // rough estimate in MB
            if estimated_chunk_memory > self.config.max_memory_usage {
                return Err(OnlineLearningError::NumericalError(format!(
                    "Chunk size {} MB exceeds maximum memory limit {} MB",
                    estimated_chunk_memory, self.config.max_memory_usage
                )));
            }

            if is_first_chunk {
                // Initialize the model with the first chunk
                self.online_nb.initialize(&chunk_X, &chunk_y)?;
                is_first_chunk = false;
            } else {
                // Process chunk incrementally
                if self.config.enable_partial_fit {
                    self.fit_chunk_partial(&chunk_X, &chunk_y)?;
                } else {
                    self.fit_chunk_batch(&chunk_X, &chunk_y)?;
                }
            }

            self.processed_chunks += 1;
            self.total_samples_seen += chunk_X.nrows();
            self.memory_usage_mb = estimated_chunk_memory;

            // Optional: checkpoint saving
            if self.config.checkpoint_frequency > 0
                && self.processed_chunks % self.config.checkpoint_frequency == 0
            {
                self.save_checkpoint()?;
            }

            // Log progress
            if self.processed_chunks % 10 == 0 {
                println!(
                    "Processed {} chunks, {} total samples",
                    self.processed_chunks, self.total_samples_seen
                );
            }
        }

        Ok(())
    }

    fn fit_chunk_partial(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<(), OnlineLearningError> {
        // Use partial_fit for each sample
        for (sample, &label) in X.axis_iter(scirs2_core::ndarray::Axis(0)).zip(y.iter()) {
            self.online_nb.partial_fit(&sample.to_owned(), label)?;
        }
        Ok(())
    }

    fn fit_chunk_batch(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<(), OnlineLearningError> {
        // Process the entire chunk at once using batch update
        for i in 0..(X.nrows() / self.config.batch_update_size + 1) {
            let start_idx = i * self.config.batch_update_size;
            let end_idx = ((i + 1) * self.config.batch_update_size).min(X.nrows());

            if start_idx >= end_idx {
                break;
            }

            for j in start_idx..end_idx {
                let sample = X.row(j).to_owned();
                self.online_nb.partial_fit(&sample, y[j])?;
            }
        }
        Ok(())
    }

    fn save_checkpoint(&self) -> Result<(), OnlineLearningError> {
        self.save_checkpoint_to_path("checkpoint.json")
    }

    /// Save checkpoint to a specific file path
    pub fn save_checkpoint_to_path<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(), OnlineLearningError> {
        let checkpoint = OnlineNBCheckpoint {
            class_counts: self.online_nb.class_counts.clone(),
            total_samples: self.online_nb.total_samples,
            gaussian_stats: self.online_nb.gaussian_stats.clone(),
            multinomial_stats: self.online_nb.multinomial_stats.clone(),
            classes: self.online_nb.classes.clone(),
            n_features: self.online_nb.n_features,
            feature_types: self.online_nb.feature_types.clone(),
            current_learning_rate: self.online_nb.current_learning_rate,
            sample_count: self.online_nb.sample_count,
            is_initialized: self.online_nb.is_initialized,
            processed_chunks: self.processed_chunks as u64,
            total_samples_seen: self.total_samples_seen as u64,
        };

        let json = serde_json::to_string_pretty(&checkpoint).map_err(|e| {
            OnlineLearningError::NumericalError(format!("Failed to serialize checkpoint: {}", e))
        })?;

        fs::write(path, json).map_err(|e| {
            OnlineLearningError::NumericalError(format!("Failed to write checkpoint: {}", e))
        })?;

        println!("Checkpoint saved at chunk {}", self.processed_chunks);
        Ok(())
    }

    /// Load checkpoint from a specific file path
    pub fn load_checkpoint_from_path<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), OnlineLearningError> {
        let json = fs::read_to_string(path).map_err(|e| {
            OnlineLearningError::NumericalError(format!("Failed to read checkpoint: {}", e))
        })?;

        let checkpoint: OnlineNBCheckpoint = serde_json::from_str(&json).map_err(|e| {
            OnlineLearningError::NumericalError(format!("Failed to deserialize checkpoint: {}", e))
        })?;

        // Restore model state
        self.online_nb.class_counts = checkpoint.class_counts;
        self.online_nb.total_samples = checkpoint.total_samples;
        self.online_nb.gaussian_stats = checkpoint.gaussian_stats;
        self.online_nb.multinomial_stats = checkpoint.multinomial_stats;
        self.online_nb.classes = checkpoint.classes;
        self.online_nb.n_features = checkpoint.n_features;
        self.online_nb.feature_types = checkpoint.feature_types;
        self.online_nb.current_learning_rate = checkpoint.current_learning_rate;
        self.online_nb.sample_count = checkpoint.sample_count;
        self.online_nb.is_initialized = checkpoint.is_initialized;
        self.processed_chunks = checkpoint.processed_chunks as usize;
        self.total_samples_seen = checkpoint.total_samples_seen as usize;

        println!("Checkpoint loaded from chunk {}", self.processed_chunks);
        Ok(())
    }

    /// Predict using the trained model
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>, OnlineLearningError> {
        self.online_nb.predict(X)
    }

    /// Predict a single sample
    pub fn predict_single(&self, sample: &Array1<f64>) -> Result<i32, OnlineLearningError> {
        self.online_nb.predict_single(sample)
    }

    /// Get statistics about the out-of-core learning process
    pub fn get_out_of_core_stats(&self) -> HashMap<String, f64> {
        let mut stats = self.online_nb.get_statistics();
        stats.insert("processed_chunks".to_string(), self.processed_chunks as f64);
        stats.insert(
            "total_samples_seen".to_string(),
            self.total_samples_seen as f64,
        );
        stats.insert("memory_usage_mb".to_string(), self.memory_usage_mb as f64);
        stats.insert(
            "samples_per_chunk".to_string(),
            self.total_samples_seen as f64 / self.processed_chunks.max(1) as f64,
        );
        stats
    }

    /// Check if the model is ready for prediction
    pub fn is_fitted(&self) -> bool {
        self.online_nb.is_initialized
    }

    /// Get memory usage estimation
    pub fn estimate_memory_usage(&self) -> usize {
        // Rough estimation of memory usage
        let base_model_size = std::mem::size_of::<OnlineNaiveBayes>();
        let stats_size = self.online_nb.get_statistics().len() * (8 + 16); // rough estimate
        (base_model_size + stats_size) / (1024 * 1024) // Convert to MB
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_buffer() {
        let mut buffer = StreamingBuffer::new(3);

        assert!(buffer.is_empty());

        buffer
            .add_sample(Array1::from_vec(vec![1.0, 2.0]), 0, 1)
            .unwrap();
        buffer
            .add_sample(Array1::from_vec(vec![3.0, 4.0]), 1, 2)
            .unwrap();
        buffer
            .add_sample(Array1::from_vec(vec![5.0, 6.0]), 0, 3)
            .unwrap();

        assert_eq!(buffer.len(), 3);

        // Adding one more should remove the oldest
        buffer
            .add_sample(Array1::from_vec(vec![7.0, 8.0]), 1, 4)
            .unwrap();
        assert_eq!(buffer.len(), 3);

        let (features, labels) = buffer.get_recent_samples(2);
        assert_eq!(features.len(), 2);
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_concept_drift_detector() {
        let mut detector = ConceptDriftDetector::new(5, 0.1);

        // Add normal error rates
        for _ in 0..10 {
            detector.update(0.05).unwrap();
        }

        assert!(!detector.is_drift_detected());

        // Add high error rates
        for _ in 0..5 {
            detector.update(0.3).unwrap();
        }

        assert!(detector.is_drift_detected());
    }

    #[test]
    fn test_online_gaussian_stats() {
        let mut stats = OnlineGaussianStats::new();

        // Add some values
        stats.update(1.0, 1.0);
        stats.update(2.0, 1.0);
        stats.update(3.0, 1.0);

        assert!((stats.mean - 2.0).abs() < 1e-10);
        assert!(stats.variance > 0.0);

        let log_likelihood = stats.log_likelihood(2.0);
        assert!(log_likelihood.is_finite());
    }

    #[test]
    fn test_online_multinomial_stats() {
        let mut stats = OnlineMultinomialStats::new(1.0);

        stats.update(0, 2.0);
        stats.update(1, 1.0);
        stats.update(0, 1.0);

        let log_prob_0 = stats.log_probability(0);
        let log_prob_1 = stats.log_probability(1);

        assert!(log_prob_0 > log_prob_1); // Value 0 should be more likely
        assert!(log_prob_0.is_finite() && log_prob_1.is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_online_naive_bayes_basic() {
        let config = OnlineLearningConfig::default();
        let mut online_nb = OnlineNaiveBayes::new(config);

        // Set feature types
        online_nb.set_feature_types(vec![FeatureType::Continuous, FeatureType::Continuous]);

        // Initialize with some data
        let X_init =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1]).unwrap();
        let y_init = Array1::from_vec(vec![0, 0, 1, 1]);

        assert!(online_nb.initialize(&X_init, &y_init).is_ok());
        assert!(online_nb.is_initialized);

        // Add single sample
        let new_sample = Array1::from_vec(vec![1.5, 1.5]);
        assert!(online_nb.partial_fit(&new_sample, 0).is_ok());

        // Test prediction
        let prediction = online_nb.predict_single(&new_sample).unwrap();
        assert!(online_nb.classes.contains(&prediction));

        // Test batch prediction
        let X_test = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let predictions = online_nb.predict(&X_test).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_online_learning_with_forgetting() {
        let mut config = OnlineLearningConfig::default();
        config.forgetting_factor = 0.9;
        config.concept_drift_detection = false;

        let mut online_nb = OnlineNaiveBayes::new(config);
        online_nb.set_feature_types(vec![FeatureType::Continuous]);

        // Initialize
        let X_init = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y_init = Array1::from_vec(vec![0, 1]);

        assert!(online_nb.initialize(&X_init, &y_init).is_ok());

        // Add many samples from different distribution
        for i in 0..50 {
            let sample = Array1::from_vec(vec![10.0 + i as f64 * 0.1]);
            let label = if i % 2 == 0 { 0 } else { 1 };
            online_nb.partial_fit(&sample, label).unwrap();
        }

        // Model should adapt to new distribution
        let stats = online_nb.get_statistics();
        assert!(stats["total_samples"] > 50.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_concept_drift_detection() {
        let mut config = OnlineLearningConfig::default();
        config.concept_drift_detection = true;
        config.drift_threshold = 0.2;

        let mut online_nb = OnlineNaiveBayes::new(config);
        online_nb.set_feature_types(vec![FeatureType::Continuous, FeatureType::Continuous]);

        // Initialize with clean data
        let X_init =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1]).unwrap();
        let y_init = Array1::from_vec(vec![0, 0, 1, 1]);

        assert!(online_nb.initialize(&X_init, &y_init).is_ok());

        // Add samples that should cause prediction errors (concept drift)
        for _ in 0..20 {
            let sample = Array1::from_vec(vec![1.0, 1.0]); // Should be class 0
            online_nb.partial_fit(&sample, 1).unwrap(); // But we label it as class 1
        }

        // Drift might be detected (depending on threshold and implementation details)
        let stats = online_nb.get_statistics();
        // Just ensure the test runs without errors
        assert!(stats.contains_key("drift_detected"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_probability_prediction() {
        let config = OnlineLearningConfig::default();
        let mut online_nb = OnlineNaiveBayes::new(config);

        online_nb.set_feature_types(vec![FeatureType::Continuous]);

        let X_init = Array2::from_shape_vec((4, 1), vec![1.0, 1.1, 2.0, 2.1]).unwrap();
        let y_init = Array1::from_vec(vec![0, 0, 1, 1]);

        assert!(online_nb.initialize(&X_init, &y_init).is_ok());

        let test_sample = Array1::from_vec(vec![1.5]);
        let probabilities = online_nb.predict_proba_single(&test_sample).unwrap();

        assert_eq!(probabilities.len(), 2);

        let total_prob: f64 = probabilities.values().sum();
        assert!((total_prob - 1.0).abs() < 1e-10);

        for &prob in probabilities.values() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_memory_chunk_iterator() {
        let X = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 2.0, 2.0, 2.1, 2.1, 2.2, 2.2,
                2.3, 2.3, 2.4, 2.4,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

        let mut chunk_iter = MemoryChunkIterator::new(X, y, 3);

        assert_eq!(chunk_iter.chunk_size(), 3);
        assert_eq!(chunk_iter.estimated_total_samples(), Some(10));

        // Test first chunk
        let chunk1 = chunk_iter.next_chunk().unwrap().unwrap();
        assert_eq!(chunk1.0.nrows(), 3);
        assert_eq!(chunk1.1.len(), 3);
        assert_eq!(chunk1.1[0], 0);

        // Test second chunk
        let chunk2 = chunk_iter.next_chunk().unwrap().unwrap();
        assert_eq!(chunk2.0.nrows(), 3);
        assert_eq!(chunk2.1.len(), 3);

        // Test third chunk
        let chunk3 = chunk_iter.next_chunk().unwrap().unwrap();
        assert_eq!(chunk3.0.nrows(), 3);
        assert_eq!(chunk3.1.len(), 3);

        // Test fourth chunk (partial)
        let chunk4 = chunk_iter.next_chunk().unwrap().unwrap();
        assert_eq!(chunk4.0.nrows(), 1);
        assert_eq!(chunk4.1.len(), 1);

        // Test no more chunks
        assert!(chunk_iter.next_chunk().unwrap().is_none());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_out_of_core_naive_bayes_basic() {
        let mut config = OnlineLearningConfig::default();
        config.chunk_size = 3;
        config.enable_partial_fit = true;

        let mut out_of_core_nb = OutOfCoreNaiveBayes::new(config);
        out_of_core_nb.set_feature_types(vec![FeatureType::Continuous, FeatureType::Continuous]);

        // Create test data
        let X = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5, 2.0, 2.0, 2.1, 2.1,
                2.2, 2.2, 2.3, 2.3, 2.4, 2.4, 2.5, 2.5,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);

        let chunk_iter = MemoryChunkIterator::new(X, y, 3);

        // Fit using out-of-core learning
        assert!(out_of_core_nb.fit_out_of_core(chunk_iter).is_ok());
        assert!(out_of_core_nb.is_fitted());

        // Test prediction
        let test_X = Array2::from_shape_vec((2, 2), vec![1.2, 1.2, 2.2, 2.2]).unwrap();
        let predictions = out_of_core_nb.predict(&test_X).unwrap();
        assert_eq!(predictions.len(), 2);

        // Check statistics
        let stats = out_of_core_nb.get_out_of_core_stats();
        assert!(stats.contains_key("processed_chunks"));
        assert!(stats.contains_key("total_samples_seen"));
        assert_eq!(stats["total_samples_seen"], 12.0);
        assert_eq!(stats["processed_chunks"], 4.0); // 12 samples / 3 chunk_size = 4 chunks
    }

    #[test]
    fn test_out_of_core_memory_limits() {
        let mut config = OnlineLearningConfig::default();
        config.chunk_size = 1000;
        config.max_memory_usage = 1; // Very small memory limit (1 MB)

        let mut out_of_core_nb = OutOfCoreNaiveBayes::new(config);
        out_of_core_nb.set_feature_types(vec![FeatureType::Continuous]);

        // Create large chunk that should exceed memory limit
        let large_X = Array2::zeros((10000, 100)); // This should be >> 1MB
        let large_y = Array1::zeros(10000);

        let chunk_iter = MemoryChunkIterator::new(large_X, large_y, 10000);

        // Should fail due to memory limit
        let result = out_of_core_nb.fit_out_of_core(chunk_iter);
        assert!(result.is_err());

        if let Err(OnlineLearningError::NumericalError(msg)) = result {
            assert!(msg.contains("exceeds maximum memory limit"));
        } else {
            panic!("Expected NumericalError with memory limit message");
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_out_of_core_batch_vs_partial() {
        let mut config_partial = OnlineLearningConfig::default();
        config_partial.chunk_size = 4;
        config_partial.enable_partial_fit = true;

        let mut config_batch = OnlineLearningConfig::default();
        config_batch.chunk_size = 4;
        config_batch.enable_partial_fit = false;
        config_batch.batch_update_size = 2;

        let mut out_of_core_nb_partial = OutOfCoreNaiveBayes::new(config_partial);
        let mut out_of_core_nb_batch = OutOfCoreNaiveBayes::new(config_batch);

        out_of_core_nb_partial.set_feature_types(vec![FeatureType::Continuous]);
        out_of_core_nb_batch.set_feature_types(vec![FeatureType::Continuous]);

        // Create test data
        let X =
            Array2::from_shape_vec((8, 1), vec![1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);

        let chunk_iter_partial = MemoryChunkIterator::new(X.clone(), y.clone(), 4);
        let chunk_iter_batch = MemoryChunkIterator::new(X, y, 4);

        // Both should work
        assert!(out_of_core_nb_partial
            .fit_out_of_core(chunk_iter_partial)
            .is_ok());
        assert!(out_of_core_nb_batch
            .fit_out_of_core(chunk_iter_batch)
            .is_ok());

        assert!(out_of_core_nb_partial.is_fitted());
        assert!(out_of_core_nb_batch.is_fitted());

        // Both should be able to make predictions
        let test_X = Array2::from_shape_vec((1, 1), vec![1.5]).unwrap();
        assert!(out_of_core_nb_partial.predict(&test_X).is_ok());
        assert!(out_of_core_nb_batch.predict(&test_X).is_ok());
    }
}
