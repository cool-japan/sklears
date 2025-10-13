//! Memory-efficient ensemble methods for large-scale machine learning
//!
//! This module provides memory-efficient implementations of ensemble methods
//! that can handle large datasets by using incremental learning, streaming
//! processing, and lazy evaluation techniques.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::Predict;
use sklears_core::traits::{Estimator, Fit, Trained, Untrained};
use sklears_core::types::Float;
use std::collections::VecDeque;
use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for memory-efficient ensemble methods
#[derive(Debug, Clone)]
pub struct MemoryEfficientConfig {
    /// Maximum number of estimators to keep in memory
    pub max_estimators_in_memory: usize,
    /// Batch size for incremental learning
    pub batch_size: usize,
    /// Window size for streaming data
    pub window_size: Option<usize>,
    /// Enable lazy evaluation for predictions
    pub lazy_evaluation: bool,
    /// Memory threshold in MB before triggering cleanup
    pub memory_threshold_mb: usize,
    /// Enable compression for stored models
    pub compress_models: bool,
    /// Use disk caching for overflow models
    pub use_disk_cache: bool,
    /// Disk cache directory
    pub cache_dir: Option<String>,
    /// Learning rate decay for incremental learning
    pub learning_rate_decay: Float,
    /// Forgetting factor for streaming data
    pub forgetting_factor: Float,
    /// Enable adaptive batch sizing
    pub adaptive_batch_size: bool,
}

impl Default for MemoryEfficientConfig {
    fn default() -> Self {
        Self {
            max_estimators_in_memory: 50,
            batch_size: 1000,
            window_size: Some(10000),
            lazy_evaluation: true,
            memory_threshold_mb: 512,
            compress_models: false,
            use_disk_cache: false,
            cache_dir: None,
            learning_rate_decay: 0.999,
            forgetting_factor: 0.95,
            adaptive_batch_size: true,
        }
    }
}

/// Memory-efficient ensemble that uses incremental learning and streaming
pub struct MemoryEfficientEnsemble<State = Untrained> {
    config: MemoryEfficientConfig,
    state: PhantomData<State>,
    // In-memory models
    active_models_: Option<Vec<Box<dyn IncrementalModel>>>,
    // Model weights for ensemble voting
    model_weights_: Option<Array1<Float>>,
    // Statistics for memory management
    memory_usage_: usize,
    total_models_created_: usize,
    // Streaming data buffer
    data_buffer_: Option<VecDeque<(Array1<Float>, Float)>>,
    // Lazy evaluation cache
    prediction_cache_: Option<std::collections::HashMap<u64, Float>>,
    // Performance tracking
    performance_history_: Vec<Float>,
    current_learning_rate_: Float,
}

/// Trait for incremental learning models
pub trait IncrementalModel: Send + Sync {
    /// Incrementally update the model with new data
    fn partial_fit(&mut self, x: &Array1<Float>, y: Float) -> Result<()>;

    /// Predict a single sample
    fn predict_single(&self, x: &Array1<Float>) -> Result<Float>;

    /// Get model complexity (for memory estimation)
    fn complexity(&self) -> usize;

    /// Serialize model for disk caching
    fn serialize(&self) -> Result<Vec<u8>>;

    /// Clone the model
    fn clone_model(&self) -> Box<dyn IncrementalModel>;
}

/// Helper function for deserializing incremental models
pub fn deserialize_incremental_model(data: &[u8]) -> Result<Box<dyn IncrementalModel>> {
    IncrementalLinearRegression::deserialize(data)
}

/// Simple incremental linear regression model
#[derive(Debug, Clone)]
pub struct IncrementalLinearRegression {
    weights: Array1<Float>,
    bias: Float,
    n_features: usize,
    learning_rate: Float,
    l2_reg: Float,
}

impl IncrementalLinearRegression {
    pub fn new(n_features: usize, learning_rate: Float, l2_reg: Float) -> Self {
        Self {
            weights: Array1::zeros(n_features),
            bias: 0.0,
            n_features,
            learning_rate,
            l2_reg,
        }
    }
}

impl IncrementalModel for IncrementalLinearRegression {
    fn partial_fit(&mut self, x: &Array1<Float>, y: Float) -> Result<()> {
        if x.len() != self.n_features {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} features", x.len()),
            });
        }

        // Compute prediction
        let y_pred = self.weights.dot(x) + self.bias;

        // Compute error
        let error = y - y_pred;

        // Update weights with regularization
        for i in 0..self.n_features {
            self.weights[i] += self.learning_rate * (error * x[i] - self.l2_reg * self.weights[i]);
        }

        // Update bias
        self.bias += self.learning_rate * error;

        Ok(())
    }

    fn predict_single(&self, x: &Array1<Float>) -> Result<Float> {
        if x.len() != self.n_features {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} features", x.len()),
            });
        }

        Ok(self.weights.dot(x) + self.bias)
    }

    fn complexity(&self) -> usize {
        // Rough estimate: weights + bias + metadata
        self.n_features * 8 + 8 + 32
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        // Simple serialization for demonstration
        // In practice, you might use serde or bincode
        let mut data = Vec::new();

        // Write n_features
        data.extend_from_slice(&self.n_features.to_le_bytes());

        // Write learning_rate
        data.extend_from_slice(&self.learning_rate.to_le_bytes());

        // Write l2_reg
        data.extend_from_slice(&self.l2_reg.to_le_bytes());

        // Write bias
        data.extend_from_slice(&self.bias.to_le_bytes());

        // Write weights
        for &weight in self.weights.iter() {
            data.extend_from_slice(&weight.to_le_bytes());
        }

        Ok(data)
    }

    fn clone_model(&self) -> Box<dyn IncrementalModel> {
        Box::new(self.clone())
    }
}

impl IncrementalLinearRegression {
    pub fn deserialize(data: &[u8]) -> Result<Box<dyn IncrementalModel>> {
        if data.len() < 32 {
            return Err(SklearsError::InvalidInput(
                "Insufficient data for deserialization".to_string(),
            ));
        }

        let mut offset = 0;

        // Read n_features
        let n_features = usize::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // Read learning_rate
        let learning_rate = Float::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // Read l2_reg
        let l2_reg = Float::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // Read bias
        let bias = Float::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // Read weights
        let mut weights = Array1::zeros(n_features);
        for i in 0..n_features {
            weights[i] = Float::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offset += 8;
        }

        Ok(Box::new(IncrementalLinearRegression {
            weights,
            bias,
            n_features,
            learning_rate,
            l2_reg,
        }))
    }
}

impl<State> MemoryEfficientEnsemble<State> {
    /// Get current memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        self.memory_usage_
    }

    /// Get number of active models in memory
    pub fn active_model_count(&self) -> usize {
        self.active_models_
            .as_ref()
            .map_or(0, |models| models.len())
    }

    /// Get total number of models created
    pub fn total_models_created(&self) -> usize {
        self.total_models_created_
    }

    /// Check if memory cleanup is needed
    pub fn needs_cleanup(&self) -> bool {
        self.memory_usage_ > self.config.memory_threshold_mb * 1024 * 1024
    }
}

impl MemoryEfficientEnsemble<Untrained> {
    /// Create a new memory-efficient ensemble
    pub fn new() -> Self {
        Self {
            config: MemoryEfficientConfig::default(),
            state: PhantomData,
            active_models_: None,
            model_weights_: None,
            memory_usage_: 0,
            total_models_created_: 0,
            data_buffer_: None,
            prediction_cache_: None,
            performance_history_: Vec::new(),
            current_learning_rate_: 0.01,
        }
    }

    /// Set maximum number of estimators to keep in memory
    pub fn max_estimators_in_memory(mut self, max_estimators: usize) -> Self {
        self.config.max_estimators_in_memory = max_estimators;
        self
    }

    /// Set batch size for incremental learning
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set window size for streaming data
    pub fn window_size(mut self, window_size: Option<usize>) -> Self {
        self.config.window_size = window_size;
        self
    }

    /// Enable lazy evaluation
    pub fn lazy_evaluation(mut self, enabled: bool) -> Self {
        self.config.lazy_evaluation = enabled;
        self
    }

    /// Set memory threshold in MB
    pub fn memory_threshold_mb(mut self, threshold: usize) -> Self {
        self.config.memory_threshold_mb = threshold;
        self
    }

    /// Enable model compression
    pub fn compress_models(mut self, enabled: bool) -> Self {
        self.config.compress_models = enabled;
        self
    }

    /// Enable disk caching
    pub fn use_disk_cache(mut self, enabled: bool, cache_dir: Option<String>) -> Self {
        self.config.use_disk_cache = enabled;
        self.config.cache_dir = cache_dir;
        self
    }

    /// Set learning rate decay
    pub fn learning_rate_decay(mut self, decay: Float) -> Self {
        self.config.learning_rate_decay = decay;
        self
    }

    /// Set forgetting factor for streaming
    pub fn forgetting_factor(mut self, factor: Float) -> Self {
        self.config.forgetting_factor = factor;
        self
    }

    /// Enable adaptive batch sizing
    pub fn adaptive_batch_size(mut self, enabled: bool) -> Self {
        self.config.adaptive_batch_size = enabled;
        self
    }

    /// Create a memory-efficient ensemble with optimal settings for large datasets
    pub fn for_large_datasets() -> Self {
        Self::new()
            .max_estimators_in_memory(20)
            .batch_size(5000)
            .window_size(Some(50000))
            .lazy_evaluation(true)
            .memory_threshold_mb(1024)
            .compress_models(true)
            .use_disk_cache(true, Some("/tmp/sklears_cache".to_string()))
            .learning_rate_decay(0.995)
            .adaptive_batch_size(true)
    }

    /// Create a streaming ensemble with optimal settings
    pub fn for_streaming() -> Self {
        Self::new()
            .max_estimators_in_memory(10)
            .batch_size(100)
            .window_size(Some(5000))
            .lazy_evaluation(true)
            .memory_threshold_mb(256)
            .forgetting_factor(0.9)
            .adaptive_batch_size(true)
    }
}

impl MemoryEfficientEnsemble<Trained> {
    /// Predict with lazy evaluation
    pub fn predict_lazy(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if !self.config.lazy_evaluation {
            return self.predict(x);
        }

        let mut predictions = Array1::zeros(x.nrows());

        // Use cached predictions if available
        if let Some(cache) = &self.prediction_cache_ {
            for (i, row) in x.axis_iter(Axis(0)).enumerate() {
                let hash = self.hash_input(&row.to_owned())?;
                if let Some(&cached_pred) = cache.get(&hash) {
                    predictions[i] = cached_pred;
                } else {
                    predictions[i] = self.predict_single_internal(&row.to_owned())?;
                }
            }
        } else {
            for (i, row) in x.axis_iter(Axis(0)).enumerate() {
                predictions[i] = self.predict_single_internal(&row.to_owned())?;
            }
        }

        Ok(predictions)
    }

    /// Predict a single sample with ensemble voting
    fn predict_single_internal(&self, x: &Array1<Float>) -> Result<Float> {
        if let Some(models) = &self.active_models_ {
            if models.is_empty() {
                return Err(SklearsError::NotFitted {
                    operation: "prediction".to_string(),
                });
            }

            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for (i, model) in models.iter().enumerate() {
                let prediction = model.predict_single(x)?;
                let weight = self.model_weights_.as_ref().map(|w| w[i]).unwrap_or(1.0);

                weighted_sum += prediction * weight;
                total_weight += weight;
            }

            if total_weight > 0.0 {
                Ok(weighted_sum / total_weight)
            } else {
                Ok(0.0)
            }
        } else {
            Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            })
        }
    }

    /// Update the ensemble with new streaming data
    pub fn partial_fit(&mut self, x: &Array1<Float>, y: Float) -> Result<()> {
        // Add to buffer
        if let Some(buffer) = &mut self.data_buffer_ {
            buffer.push_back((x.clone(), y));

            // Maintain window size
            if let Some(window_size) = self.config.window_size {
                while buffer.len() > window_size {
                    buffer.pop_front();
                }
            }
        } else {
            self.data_buffer_ = Some(VecDeque::new());
            self.data_buffer_
                .as_mut()
                .unwrap()
                .push_back((x.clone(), y));
        }

        // Update active models
        if let Some(models) = &mut self.active_models_ {
            for model in models.iter_mut() {
                model.partial_fit(x, y)?;
            }
        }

        // Check if we need to add a new model
        if self.should_add_model() {
            self.add_new_model(x.len())?;
        }

        // Update learning rate
        self.current_learning_rate_ *= self.config.learning_rate_decay;

        // Check for memory cleanup
        if self.needs_cleanup() {
            self.cleanup_memory()?;
        }

        Ok(())
    }

    /// Check if a new model should be added
    fn should_add_model(&self) -> bool {
        // Add model if performance has degraded or we don't have enough diversity
        if self.performance_history_.len() < 10 {
            return true;
        }

        let recent_performance = self
            .performance_history_
            .iter()
            .rev()
            .take(5)
            .sum::<Float>()
            / 5.0;
        let older_performance = self
            .performance_history_
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .sum::<Float>()
            / 5.0;

        recent_performance < older_performance * 0.95 // 5% degradation threshold
    }

    /// Add a new incremental model to the ensemble
    fn add_new_model(&mut self, n_features: usize) -> Result<()> {
        if let Some(models) = &mut self.active_models_ {
            // Create new model
            let new_model = Box::new(IncrementalLinearRegression::new(
                n_features,
                self.current_learning_rate_,
                0.001, // L2 regularization
            ));

            models.push(new_model as Box<dyn IncrementalModel>);
            self.total_models_created_ += 1;

            // Update memory usage
            self.memory_usage_ += n_features * 8 + 64; // Rough estimate

            // Remove oldest model if we exceed memory limit
            if models.len() > self.config.max_estimators_in_memory && !models.is_empty() {
                let removed_model = models.remove(0);
                self.memory_usage_ = self
                    .memory_usage_
                    .saturating_sub(removed_model.complexity());
            }

            // Update weights
            self.update_model_weights()?;
        } else {
            // Initialize ensemble
            let mut models = Vec::new();
            let new_model = Box::new(IncrementalLinearRegression::new(
                n_features,
                self.current_learning_rate_,
                0.001,
            ));
            models.push(new_model as Box<dyn IncrementalModel>);
            self.active_models_ = Some(models);
            self.total_models_created_ = 1;
            self.memory_usage_ = n_features * 8 + 64;
            self.update_model_weights()?;
        }

        Ok(())
    }

    /// Update model weights based on performance
    fn update_model_weights(&mut self) -> Result<()> {
        if let Some(models) = &self.active_models_ {
            let n_models = models.len();
            if n_models == 0 {
                return Ok(());
            }

            // Simple equal weighting for now
            // In practice, you'd use performance-based weighting
            let weights = Array1::from_elem(n_models, 1.0 / n_models as Float);
            self.model_weights_ = Some(weights);
        }

        Ok(())
    }

    /// Cleanup memory by removing least important models
    fn cleanup_memory(&mut self) -> Result<()> {
        if let Some(models) = &mut self.active_models_ {
            let target_size = self.config.max_estimators_in_memory / 2;

            while models.len() > target_size {
                if !models.is_empty() {
                    let removed_model = models.remove(0);
                    self.memory_usage_ = self
                        .memory_usage_
                        .saturating_sub(removed_model.complexity());
                }
            }

            // Update weights after cleanup
            self.update_model_weights()?;
        }

        // Clear prediction cache
        if let Some(cache) = &mut self.prediction_cache_ {
            cache.clear();
        }

        Ok(())
    }

    /// Hash input for caching
    fn hash_input(&self, x: &Array1<Float>) -> Result<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for &val in x.iter() {
            val.to_bits().hash(&mut hasher);
        }
        Ok(hasher.finish())
    }
}

// Implement core traits
impl Estimator for MemoryEfficientEnsemble<Untrained> {
    type Config = MemoryEfficientConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for MemoryEfficientEnsemble<Untrained> {
    type Fitted = MemoryEfficientEnsemble<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", n_samples),
                actual: format!("{} samples", y.len()),
            });
        }

        // Initialize ensemble
        let config = self.config.clone();
        let mut ensemble = MemoryEfficientEnsemble::<Trained> {
            config: config.clone(),
            state: PhantomData,
            active_models_: Some(Vec::new()),
            model_weights_: None,
            memory_usage_: 0,
            total_models_created_: 0,
            data_buffer_: Some(VecDeque::new()),
            prediction_cache_: if self.config.lazy_evaluation {
                Some(std::collections::HashMap::new())
            } else {
                None
            },
            performance_history_: Vec::new(),
            current_learning_rate_: 0.01,
        };

        // Process data in batches
        let batch_size = config.batch_size;
        let mut current_batch_size = batch_size;

        for start_idx in (0..n_samples).step_by(current_batch_size) {
            let end_idx = (start_idx + current_batch_size).min(n_samples);

            // Process batch
            for i in start_idx..end_idx {
                let x_sample = x.row(i).to_owned();
                let y_sample = y[i];

                // Add first model if needed
                if ensemble.active_models_.as_ref().unwrap().is_empty() {
                    ensemble.add_new_model(n_features)?;
                }

                // Partial fit
                ensemble.partial_fit(&x_sample, y_sample)?;
            }

            // Adaptive batch size adjustment
            if config.adaptive_batch_size {
                if ensemble.memory_usage_ > config.memory_threshold_mb * 1024 * 1024 / 2 {
                    current_batch_size /= 2;
                } else if ensemble.memory_usage_ < config.memory_threshold_mb * 1024 * 1024 / 4 {
                    current_batch_size = (current_batch_size * 3 / 2).min(batch_size * 2);
                }
                current_batch_size = current_batch_size.max(100); // Minimum batch size
            }
        }

        Ok(ensemble)
    }
}

impl Predict<Array2<Float>, Array1<Float>> for MemoryEfficientEnsemble<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            predictions[i] = self.predict_single_internal(&row.to_owned())?;
        }

        Ok(predictions)
    }
}

impl Default for MemoryEfficientEnsemble<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_memory_efficient_ensemble_basic() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
                9.0, 10.0, 10.0, 11.0,
            ],
        )
        .unwrap();

        let y = array![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0];

        let ensemble = MemoryEfficientEnsemble::new()
            .max_estimators_in_memory(5)
            .batch_size(3);

        let trained = ensemble.fit(&x, &y).unwrap();

        assert!(trained.active_model_count() > 0);
        assert!(trained.memory_usage() > 0);

        let predictions = trained.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_incremental_learning() {
        let mut ensemble = MemoryEfficientEnsemble::new()
            .max_estimators_in_memory(3)
            .batch_size(2);

        // Initial training
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let mut trained = ensemble.fit(&x, &y).unwrap();

        // Incremental updates
        let x_new = array![5.0, 6.0];
        trained.partial_fit(&x_new, 11.0).unwrap();

        assert!(trained.active_model_count() > 0);

        // Test prediction
        let test_x = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let predictions = trained.predict(&test_x).unwrap();
        assert_eq!(predictions.len(), 1);
    }

    #[test]
    fn test_memory_management() {
        let ensemble = MemoryEfficientEnsemble::new()
            .max_estimators_in_memory(2)
            .memory_threshold_mb(1); // Very low threshold

        let x = Array2::from_shape_vec((20, 5), (0..100).map(|i| i as Float).collect()).unwrap();
        let y = Array1::from_shape_vec(20, (0..20).map(|i| i as Float).collect()).unwrap();

        let trained = ensemble.fit(&x, &y).unwrap();

        // Should have limited number of models due to memory constraints
        assert!(trained.active_model_count() <= 2);
        assert!(trained.total_models_created() >= trained.active_model_count());
    }

    #[test]
    fn test_lazy_evaluation() {
        let ensemble = MemoryEfficientEnsemble::new()
            .lazy_evaluation(true)
            .max_estimators_in_memory(3);

        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
        )
        .unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];

        let trained = ensemble.fit(&x, &y).unwrap();

        // Test lazy prediction
        let predictions = trained.predict_lazy(&x).unwrap();
        assert_eq!(predictions.len(), x.nrows());

        // Test regular prediction for comparison
        let regular_predictions = trained.predict(&x).unwrap();
        assert_eq!(regular_predictions.len(), x.nrows());
    }
}
