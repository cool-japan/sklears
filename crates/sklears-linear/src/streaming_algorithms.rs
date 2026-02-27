//! Streaming algorithms for large datasets
//!
//! This module provides streaming/online learning algorithms that can process
//! large datasets that don't fit in memory by processing data in chunks.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for streaming algorithms
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of data chunks to process at once
    pub chunk_size: usize,
    /// Learning rate for online updates
    pub learning_rate: Float,
    /// Learning rate decay factor
    pub learning_rate_decay: Float,
    /// Minimum learning rate
    pub min_learning_rate: Float,
    /// L1 regularization parameter
    pub alpha: Float,
    /// L2 regularization parameter
    pub l2_reg: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Maximum number of passes through the data
    pub max_epochs: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Whether to shuffle data between epochs
    pub shuffle: bool,
    /// Random seed for shuffling
    pub random_state: Option<u64>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            learning_rate: 0.01,
            learning_rate_decay: 0.99,
            min_learning_rate: 1e-6,
            alpha: 0.01,
            l2_reg: 0.01,
            fit_intercept: true,
            max_epochs: 100,
            tol: 1e-6,
            shuffle: true,
            random_state: None,
        }
    }
}

/// Streaming linear regression using stochastic gradient descent
#[derive(Debug, Clone)]
pub struct StreamingLinearRegression<State = Untrained> {
    config: StreamingConfig,
    state: PhantomData<State>,
    // Model parameters
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    // Training metadata
    n_features_: Option<usize>,
    n_samples_seen_: Option<usize>,
    current_lr_: Option<Float>,
    loss_history_: Option<Vec<Float>>,
}

impl StreamingLinearRegression<Untrained> {
    /// Create a new streaming linear regression model
    pub fn new() -> Self {
        Self {
            config: StreamingConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            n_samples_seen_: None,
            current_lr_: None,
            loss_history_: None,
        }
    }

    /// Set chunk size for streaming
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set regularization parameters
    pub fn regularization(mut self, alpha: Float, l2_reg: Float) -> Self {
        self.config.alpha = alpha;
        self.config.l2_reg = l2_reg;
        self
    }

    /// Builder pattern
    pub fn builder() -> StreamingLinearRegressionBuilder {
        StreamingLinearRegressionBuilder::default()
    }
}

/// Builder for StreamingLinearRegression
#[derive(Debug, Default)]
pub struct StreamingLinearRegressionBuilder {
    config: StreamingConfig,
}

impl StreamingLinearRegressionBuilder {
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    pub fn learning_rate_decay(mut self, decay: Float) -> Self {
        self.config.learning_rate_decay = decay;
        self
    }

    pub fn regularization(mut self, alpha: Float, l2_reg: Float) -> Self {
        self.config.alpha = alpha;
        self.config.l2_reg = l2_reg;
        self
    }

    pub fn max_epochs(mut self, epochs: usize) -> Self {
        self.config.max_epochs = epochs;
        self
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    pub fn build(self) -> StreamingLinearRegression<Untrained> {
        StreamingLinearRegression {
            config: self.config,
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            n_samples_seen_: None,
            current_lr_: None,
            loss_history_: None,
        }
    }
}

impl Default for StreamingLinearRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Estimator<State> for StreamingLinearRegression<State> {
    type Float = Float;
    type Config = StreamingConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for StreamingLinearRegression<Untrained> {
    type Fitted = StreamingLinearRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.nrows() == y.len()".to_string(),
                actual: format!("X.nrows()={}, y.len()={}", n_samples, y.len()),
            });
        }

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        // Initialize parameters
        let mut coef = Array1::zeros(n_features);
        let mut intercept = 0.0;
        let mut current_lr = self.config.learning_rate;
        let mut loss_history = Vec::new();

        // Precompute indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Training loop
        for epoch in 0..self.config.max_epochs {
            // Shuffle data if requested
            if self.config.shuffle {
                self.shuffle_indices(&mut indices, epoch);
            }

            let mut epoch_loss = 0.0;
            let mut n_chunks = 0;

            // Process data in chunks
            for chunk_start in (0..n_samples).step_by(self.config.chunk_size) {
                let chunk_end = (chunk_start + self.config.chunk_size).min(n_samples);

                // Get chunk indices
                let chunk_indices = &indices[chunk_start..chunk_end];

                // Extract chunk data
                let (x_chunk, y_chunk) = self.extract_chunk(x, y, chunk_indices)?;

                // Update parameters with this chunk
                let chunk_loss = self.update_parameters(
                    &x_chunk,
                    &y_chunk,
                    &mut coef,
                    &mut intercept,
                    current_lr,
                )?;

                epoch_loss += chunk_loss;
                n_chunks += 1;
            }

            // Average loss for this epoch
            let avg_epoch_loss = epoch_loss / n_chunks as Float;
            loss_history.push(avg_epoch_loss);

            // Check convergence
            if epoch > 0 {
                let loss_change = (loss_history[epoch] - loss_history[epoch - 1]).abs();
                if loss_change < self.config.tol {
                    break;
                }
            }

            // Update learning rate
            current_lr *= self.config.learning_rate_decay;
            current_lr = current_lr.max(self.config.min_learning_rate);
        }

        Ok(StreamingLinearRegression {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            n_features_: Some(n_features),
            n_samples_seen_: Some(n_samples),
            current_lr_: Some(current_lr),
            loss_history_: Some(loss_history),
        })
    }
}

impl StreamingLinearRegression<Untrained> {
    /// Extract a chunk of data based on indices
    fn extract_chunk(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        indices: &[usize],
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let n_features = x.ncols();
        let chunk_size = indices.len();

        let mut x_chunk = Array2::zeros((chunk_size, n_features));
        let mut y_chunk = Array1::zeros(chunk_size);

        for (i, &idx) in indices.iter().enumerate() {
            x_chunk.row_mut(i).assign(&x.row(idx));
            y_chunk[i] = y[idx];
        }

        Ok((x_chunk, y_chunk))
    }

    /// Update model parameters using a chunk of data
    fn update_parameters(
        &self,
        x_chunk: &Array2<Float>,
        y_chunk: &Array1<Float>,
        coef: &mut Array1<Float>,
        intercept: &mut Float,
        learning_rate: Float,
    ) -> Result<Float> {
        let n_samples = x_chunk.nrows();
        let n_features = x_chunk.ncols();

        // Forward pass: compute predictions and loss
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            predictions[i] = x_chunk.row(i).dot(coef) + *intercept;
        }

        // Compute residuals and loss
        let residuals = &predictions - y_chunk;
        let loss = residuals.mapv(|x| x * x).sum() / (2.0 * n_samples as Float);

        // Add regularization to loss
        let reg_loss = loss
            + self.config.alpha * coef.mapv(Float::abs).sum()
            + 0.5 * self.config.l2_reg * coef.mapv(|x| x * x).sum();

        // Compute gradients
        let mut coef_gradient: Array1<Float> = Array1::zeros(n_features);
        let mut intercept_gradient = 0.0;

        for i in 0..n_samples {
            let residual = residuals[i];

            // Coefficient gradients
            for j in 0..n_features {
                coef_gradient[j] += residual * x_chunk[[i, j]] / n_samples as Float;
            }

            // Intercept gradient
            if self.config.fit_intercept {
                intercept_gradient += residual / n_samples as Float;
            }
        }

        // Add regularization gradients
        for j in 0..n_features {
            // L2 regularization
            coef_gradient[j] += self.config.l2_reg * coef[j];

            // L1 regularization (subgradient)
            if coef[j] > 0.0 {
                coef_gradient[j] += self.config.alpha;
            } else if coef[j] < 0.0 {
                coef_gradient[j] -= self.config.alpha;
            }
        }

        // Update parameters
        for j in 0..n_features {
            coef[j] -= (learning_rate * coef_gradient[j]) as Float;
        }

        if self.config.fit_intercept {
            *intercept -= learning_rate * intercept_gradient;
        }

        Ok(reg_loss)
    }

    /// Shuffle indices for data randomization
    fn shuffle_indices(&self, indices: &mut [usize], seed: usize) {
        // Simple shuffle implementation using seed for reproducibility
        let n = indices.len();
        for i in (1..n).rev() {
            let j = (seed * 1664525 + 1013904223 + i) % (i + 1);
            indices.swap(i, j);
        }
    }
}

impl Predict<Array2<Float>, Array1<Float>> for StreamingLinearRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: n_features not available".to_string())
        })?;

        if x.ncols() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: n_features,
                actual: x.ncols(),
            });
        }

        let coef = self.coef_.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: coefficients not available".to_string())
        })?;
        let intercept = self.intercept_.ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: intercept not available".to_string())
        })?;

        let predictions = x.dot(coef) + intercept;
        Ok(predictions)
    }
}

impl StreamingLinearRegression<Trained> {
    /// Get coefficients
    pub fn coef(&self) -> Result<&Array1<Float>> {
        self.coef_.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: coefficients not available".to_string())
        })
    }

    /// Get intercept
    pub fn intercept(&self) -> Result<Float> {
        self.intercept_.ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: intercept not available".to_string())
        })
    }

    /// Get number of samples seen during training
    pub fn n_samples_seen(&self) -> Result<usize> {
        self.n_samples_seen_.ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: n_samples_seen not available".to_string())
        })
    }

    /// Get loss history
    pub fn loss_history(&self) -> Result<&Vec<Float>> {
        self.loss_history_.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: loss_history not available".to_string())
        })
    }

    /// Partial fit with new data (online learning)
    pub fn partial_fit(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<()> {
        let n_features = self.n_features_.ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: n_features not available".to_string())
        })?;

        if x.ncols() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: n_features,
                actual: x.ncols(),
            });
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.nrows() == y.len()".to_string(),
                actual: format!("X.nrows()={}, y.len()={}", x.nrows(), y.len()),
            });
        }

        let mut coef = self.coef_.take().ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: coefficients not available".to_string())
        })?;
        let mut intercept = self.intercept_.ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: intercept not available".to_string())
        })?;
        let current_lr = self.current_lr_.ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: current_lr not available".to_string())
        })?;

        // Use a temporary config and create a temporary untrained model for update logic
        let temp_model = StreamingLinearRegression {
            config: self.config.clone(),
            state: PhantomData::<Untrained>,
            coef_: None,
            intercept_: None,
            n_features_: None,
            n_samples_seen_: None,
            current_lr_: None,
            loss_history_: None,
        };

        // Update parameters with new data
        let _loss = temp_model.update_parameters(x, y, &mut coef, &mut intercept, current_lr)?;

        // Update model state
        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
        let current_samples = self.n_samples_seen_.ok_or_else(|| {
            SklearsError::InvalidState("Model not fitted: n_samples_seen not available".to_string())
        })?;
        self.n_samples_seen_ = Some(current_samples + x.nrows());

        Ok(())
    }
}

/// Streaming Lasso regression
#[derive(Debug, Clone)]
pub struct StreamingLasso<State = Untrained> {
    base_model: StreamingLinearRegression<State>,
}

impl StreamingLasso<Untrained> {
    /// Create new streaming Lasso
    pub fn new() -> Self {
        let config = StreamingConfig {
            l2_reg: 0.0, // Pure L1 regularization
            ..StreamingConfig::default()
        };

        Self {
            base_model: StreamingLinearRegression {
                config,
                state: PhantomData,
                coef_: None,
                intercept_: None,
                n_features_: None,
                n_samples_seen_: None,
                current_lr_: None,
                loss_history_: None,
            },
        }
    }

    /// Set alpha parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.base_model.config.alpha = alpha;
        self
    }

    /// Set chunk size
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base_model.config.chunk_size = size;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.base_model.config.learning_rate = lr;
        self
    }
}

impl Default for StreamingLasso<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Estimator<State> for StreamingLasso<State> {
    type Float = Float;
    type Config = StreamingConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        self.base_model.config()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for StreamingLasso<Untrained> {
    type Fitted = StreamingLasso<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let fitted_base = self.base_model.fit(x, y)?;
        Ok(StreamingLasso {
            base_model: fitted_base,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for StreamingLasso<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        self.base_model.predict(x)
    }
}

impl StreamingLasso<Trained> {
    /// Get coefficients
    pub fn coef(&self) -> Result<&Array1<Float>> {
        self.base_model.coef()
    }

    /// Get intercept
    pub fn intercept(&self) -> Result<Float> {
        self.base_model.intercept()
    }

    /// Partial fit with new data
    pub fn partial_fit(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<()> {
        self.base_model.partial_fit(x, y)
    }
}

/// Data stream iterator for processing large datasets
pub struct DataStreamIterator<'a> {
    x: &'a Array2<Float>,
    y: &'a Array1<Float>,
    chunk_size: usize,
    current_pos: usize,
    indices: Vec<usize>,
}

impl<'a> DataStreamIterator<'a> {
    /// Create new data stream iterator
    pub fn new(
        x: &'a Array2<Float>,
        y: &'a Array1<Float>,
        chunk_size: usize,
        shuffle: bool,
    ) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.nrows() == y.len()".to_string(),
                actual: format!("X.nrows()={}, y.len()={}", x.nrows(), y.len()),
            });
        }

        let mut indices: Vec<usize> = (0..x.nrows()).collect();

        if shuffle {
            // Simple shuffle for demonstration
            for i in (1..indices.len()).rev() {
                let j = i % (i + 1);
                indices.swap(i, j);
            }
        }

        Ok(Self {
            x,
            y,
            chunk_size,
            current_pos: 0,
            indices,
        })
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }

    /// Get total number of samples
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if iterator has no samples
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Check if iterator is finished
    pub fn is_finished(&self) -> bool {
        self.current_pos >= self.indices.len()
    }
}

impl<'a> Iterator for DataStreamIterator<'a> {
    type Item = Result<(Array2<Float>, Array1<Float>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_finished() {
            return None;
        }

        let chunk_end = (self.current_pos + self.chunk_size).min(self.indices.len());
        let chunk_indices = &self.indices[self.current_pos..chunk_end];

        // Extract chunk
        let n_features = self.x.ncols();
        let chunk_size = chunk_indices.len();

        let mut x_chunk = Array2::zeros((chunk_size, n_features));
        let mut y_chunk = Array1::zeros(chunk_size);

        for (i, &idx) in chunk_indices.iter().enumerate() {
            x_chunk.row_mut(i).assign(&self.x.row(idx));
            y_chunk[i] = self.y[idx];
        }

        self.current_pos = chunk_end;

        Some(Ok((x_chunk, y_chunk)))
    }
}

/// Utilities for streaming processing
pub struct StreamingUtils;

impl StreamingUtils {
    /// Estimate memory usage for a given dataset and chunk size
    pub fn estimate_memory_usage(
        n_samples: usize,
        n_features: usize,
        chunk_size: usize,
    ) -> (usize, usize) {
        let full_dataset_size = n_samples * n_features * std::mem::size_of::<Float>();
        let chunk_size_bytes = chunk_size * n_features * std::mem::size_of::<Float>();

        (full_dataset_size, chunk_size_bytes)
    }

    /// Recommend chunk size based on available memory
    pub fn recommend_chunk_size(n_features: usize, available_memory_mb: usize) -> usize {
        let memory_bytes = available_memory_mb * 1024 * 1024;
        let safety_factor = 0.8; // Use only 80% of available memory

        let usable_memory = (memory_bytes as Float * safety_factor) as usize;
        let bytes_per_sample = n_features * std::mem::size_of::<Float>();

        (usable_memory / bytes_per_sample).max(1)
    }

    /// Process dataset in streaming fashion with custom function
    pub fn stream_process<F>(
        x: &Array2<Float>,
        y: &Array1<Float>,
        chunk_size: usize,
        mut process_fn: F,
    ) -> Result<()>
    where
        F: FnMut(&Array2<Float>, &Array1<Float>) -> Result<()>,
    {
        let stream = DataStreamIterator::new(x, y, chunk_size, false)?;

        for chunk_result in stream {
            let (x_chunk, y_chunk) = chunk_result?;
            process_fn(&x_chunk, &y_chunk)?;
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_streaming_linear_regression() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![3.0, 5.0, 7.0, 9.0]; // y = x1 + x2

        let model = StreamingLinearRegression::builder()
            .chunk_size(2)
            .learning_rate(0.1)
            .max_epochs(50)
            .build();

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are reasonable
        for i in 0..predictions.len() {
            assert!((predictions[i] - y[i]).abs() < 1.0);
        }

        assert_eq!(fitted.n_samples_seen().unwrap(), 4);
        assert!(!fitted.loss_history().unwrap().is_empty());
    }

    #[test]
    fn test_streaming_lasso() {
        let x = array![[1.0, 0.0, 2.0], [2.0, 0.0, 3.0], [3.0, 0.0, 4.0],];

        let y = array![3.0, 7.0, 11.0]; // y = x1 + 2*x3 (x2 should be zero)

        let model = StreamingLasso::new()
            .alpha(0.1)
            .chunk_size(2)
            .learning_rate(0.1);

        let fitted = model.fit(&x, &y).unwrap();
        let coef = fitted.coef().unwrap();

        // Should have non-zero coefficients for relevant features
        assert!(coef[0].abs() > 0.1); // x1 should be important
        assert!(coef[2].abs() > 0.1); // x3 should be important

        // x2 should be close to zero due to L1 regularization
        assert!(coef[1].abs() < 0.5);
    }

    #[test]
    fn test_data_stream_iterator() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let y = array![1.0, 2.0, 3.0, 4.0];

        let mut stream = DataStreamIterator::new(&x, &y, 2, false).unwrap();

        // First chunk
        let chunk1 = stream.next().unwrap().unwrap();
        assert_eq!(chunk1.0.nrows(), 2);
        assert_eq!(chunk1.1.len(), 2);

        // Second chunk
        let chunk2 = stream.next().unwrap().unwrap();
        assert_eq!(chunk2.0.nrows(), 2);
        assert_eq!(chunk2.1.len(), 2);

        // Should be finished
        assert!(stream.next().is_none());
        assert!(stream.is_finished());
    }

    #[test]
    fn test_partial_fit() {
        let x1 = array![[1.0, 2.0], [2.0, 3.0]];
        let y1 = array![3.0, 5.0];

        let model = StreamingLinearRegression::new().chunk_size(1);
        let mut fitted = model.fit(&x1, &y1).unwrap();

        let original_samples = fitted.n_samples_seen().unwrap();

        // Add more data
        let x2 = array![[3.0, 4.0], [4.0, 5.0]];
        let y2 = array![7.0, 9.0];

        fitted.partial_fit(&x2, &y2).unwrap();

        assert_eq!(fitted.n_samples_seen().unwrap(), original_samples + 2);
    }

    #[test]
    fn test_streaming_utils() {
        let (full_size, chunk_size) = StreamingUtils::estimate_memory_usage(1000, 50, 100);

        assert_eq!(full_size, 1000 * 50 * std::mem::size_of::<Float>());
        assert_eq!(chunk_size, 100 * 50 * std::mem::size_of::<Float>());

        let recommended = StreamingUtils::recommend_chunk_size(100, 100); // 100MB
        assert!(recommended > 0);
    }

    #[test]
    fn test_stream_process() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let mut chunk_count = 0;
        let result = StreamingUtils::stream_process(&x, &y, 2, |x_chunk, y_chunk| {
            chunk_count += 1;
            assert!(x_chunk.nrows() > 0);
            assert!(!y_chunk.is_empty());
            Ok(())
        });

        assert!(result.is_ok());
        assert_eq!(chunk_count, 2); // 3 samples with chunk_size=2 -> 2 chunks
    }
}
