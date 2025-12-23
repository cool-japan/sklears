//! Streaming and Incremental Learning for Multi-Output Prediction
//!
//! This module provides algorithms for learning from streaming data with multiple outputs,
//! including incremental learning, online learning, and concept drift detection.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::VecDeque;

// ============================================================================
// Incremental Multi-Output Regression
// ============================================================================

/// Configuration for incremental multi-output regression
#[derive(Debug, Clone)]
pub struct IncrementalMultiOutputRegressionConfig {
    /// Learning rate for gradient updates
    pub learning_rate: Float,
    /// L2 regularization parameter
    pub alpha: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Maximum number of samples to keep in memory (for computing statistics)
    pub max_samples: usize,
    /// Whether to use adaptive learning rate
    pub adaptive_learning_rate: bool,
    /// Decay factor for learning rate
    pub learning_rate_decay: Float,
}

impl Default for IncrementalMultiOutputRegressionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            alpha: 0.0001,
            fit_intercept: true,
            max_samples: 10000,
            adaptive_learning_rate: true,
            learning_rate_decay: 0.999,
        }
    }
}

/// Incremental Multi-Output Regressor
///
/// Online learning algorithm that can learn from data streams with multiple outputs.
/// Uses stochastic gradient descent with optional adaptive learning rates.
///
/// # Examples
///
/// ```rust
/// use sklears_multioutput::streaming::{IncrementalMultiOutputRegression, IncrementalMultiOutputRegressionConfig};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Predict};
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
///
/// let mut model = IncrementalMultiOutputRegression::new();
/// let trained = model.fit(&X.view(), &y.view()).unwrap();
///
/// // Continue learning with new data
/// let X_new = array![[4.0, 5.0]];
/// let y_new = array![[4.0, 5.0]];
/// let updated = trained.partial_fit(&X_new.view(), &y_new.view()).unwrap();
///
/// let predictions = updated.predict(&X.view()).unwrap();
/// assert_eq!(predictions.dim(), (3, 2));
/// ```
#[derive(Debug, Clone)]
pub struct IncrementalMultiOutputRegression<S = Untrained> {
    state: S,
    config: IncrementalMultiOutputRegressionConfig,
}

/// Trained state for Incremental Multi-Output Regression
#[derive(Debug, Clone)]
pub struct IncrementalMultiOutputRegressionTrained {
    /// Coefficient matrix (n_features x n_outputs)
    pub coef: Array2<Float>,
    /// Intercept vector (n_outputs)
    pub intercept: Array1<Float>,
    /// Number of features
    pub n_features: usize,
    /// Number of outputs
    pub n_outputs: usize,
    /// Number of samples seen so far
    pub n_samples_seen: usize,
    /// Current learning rate
    pub current_learning_rate: Float,
    /// Running mean of features (for normalization)
    pub feature_mean: Array1<Float>,
    /// Running std of features (for normalization)
    pub feature_std: Array1<Float>,
    /// Configuration
    pub config: IncrementalMultiOutputRegressionConfig,
}

impl IncrementalMultiOutputRegression<Untrained> {
    /// Create a new incremental multi-output regressor
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: IncrementalMultiOutputRegressionConfig::default(),
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: IncrementalMultiOutputRegressionConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }
}

impl Default for IncrementalMultiOutputRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>>
    for IncrementalMultiOutputRegression<Untrained>
{
    type Fitted = IncrementalMultiOutputRegression<IncrementalMultiOutputRegressionTrained>;

    fn fit(self, X: &ArrayView2<Float>, y: &ArrayView2<Float>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_outputs = y.ncols();

        // Initialize coefficients
        let mut coef = Array2::zeros((n_features, n_outputs));
        let mut intercept = Array1::zeros(n_outputs);

        // Compute feature statistics
        let feature_mean = X.mean_axis(Axis(0)).unwrap();
        let feature_std = X.std_axis(Axis(0), 0.0);

        let mut current_learning_rate = self.config.learning_rate;

        // Perform initial gradient descent over the batch
        for _ in 0..10 {
            // Mini-batch iterations
            for i in 0..n_samples {
                let x_i = X.row(i);
                let y_i = y.row(i);

                // Prediction
                let pred = coef.t().dot(&x_i) + &intercept;

                // Error
                let error = &y_i - &pred;

                // Update coefficients using gradient descent
                for j in 0..n_features {
                    for k in 0..n_outputs {
                        let gradient = -error[k] * x_i[j] + self.config.alpha * coef[[j, k]];
                        coef[[j, k]] -= current_learning_rate * gradient;
                    }
                }

                // Update intercept
                if self.config.fit_intercept {
                    for k in 0..n_outputs {
                        intercept[k] += current_learning_rate * error[k];
                    }
                }
            }

            // Decay learning rate
            if self.config.adaptive_learning_rate {
                current_learning_rate *= self.config.learning_rate_decay;
            }
        }

        Ok(IncrementalMultiOutputRegression {
            state: IncrementalMultiOutputRegressionTrained {
                coef,
                intercept,
                n_features,
                n_outputs,
                n_samples_seen: n_samples,
                current_learning_rate,
                feature_mean,
                feature_std,
                config: self.config,
            },
            config: IncrementalMultiOutputRegressionConfig::default(),
        })
    }
}

impl IncrementalMultiOutputRegression<IncrementalMultiOutputRegressionTrained> {
    /// Partial fit on new data (incremental learning)
    pub fn partial_fit(mut self, X: &ArrayView2<Float>, y: &ArrayView2<Float>) -> SklResult<Self> {
        if X.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                X.ncols()
            )));
        }

        if y.ncols() != self.state.n_outputs {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} outputs, got {}",
                self.state.n_outputs,
                y.ncols()
            )));
        }

        let n_samples = X.nrows();

        // Update feature statistics (running average)
        let n_old = self.state.n_samples_seen as Float;
        let n_new = n_samples as Float;
        let n_total = n_old + n_new;

        let new_mean = X.mean_axis(Axis(0)).unwrap();
        self.state.feature_mean = (&self.state.feature_mean * n_old + &new_mean * n_new) / n_total;

        // Perform incremental updates
        for i in 0..n_samples {
            let x_i = X.row(i);
            let y_i = y.row(i);

            // Prediction
            let pred = self.state.coef.t().dot(&x_i) + &self.state.intercept;

            // Error
            let error = &y_i - &pred;

            // Update coefficients
            for j in 0..self.state.n_features {
                for k in 0..self.state.n_outputs {
                    let gradient =
                        -error[k] * x_i[j] + self.state.config.alpha * self.state.coef[[j, k]];
                    self.state.coef[[j, k]] -= self.state.current_learning_rate * gradient;
                }
            }

            // Update intercept
            if self.state.config.fit_intercept {
                for k in 0..self.state.n_outputs {
                    self.state.intercept[k] += self.state.current_learning_rate * error[k];
                }
            }
        }

        // Update statistics
        self.state.n_samples_seen += n_samples;

        // Decay learning rate
        if self.state.config.adaptive_learning_rate {
            self.state.current_learning_rate *= self.state.config.learning_rate_decay;
        }

        Ok(self)
    }

    /// Get the current coefficients
    pub fn coef(&self) -> &Array2<Float> {
        &self.state.coef
    }

    /// Get the current intercept
    pub fn intercept(&self) -> &Array1<Float> {
        &self.state.intercept
    }

    /// Get number of samples seen
    pub fn n_samples_seen(&self) -> usize {
        self.state.n_samples_seen
    }

    /// Get current learning rate
    pub fn current_learning_rate(&self) -> Float {
        self.state.current_learning_rate
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>>
    for IncrementalMultiOutputRegression<IncrementalMultiOutputRegressionTrained>
{
    fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                X.ncols()
            )));
        }

        let n_samples = X.nrows();
        let mut predictions = Array2::zeros((n_samples, self.state.n_outputs));

        for i in 0..n_samples {
            let x_i = X.row(i);
            let pred = self.state.coef.t().dot(&x_i) + &self.state.intercept;
            predictions.row_mut(i).assign(&pred);
        }

        Ok(predictions)
    }
}

impl Estimator for IncrementalMultiOutputRegression<Untrained> {
    type Config = IncrementalMultiOutputRegressionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for IncrementalMultiOutputRegression<IncrementalMultiOutputRegressionTrained> {
    type Config = IncrementalMultiOutputRegressionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.state.config
    }
}

// ============================================================================
// Streaming Multi-Output with Mini-Batches
// ============================================================================

/// Configuration for streaming multi-output learning
#[derive(Debug, Clone)]
pub struct StreamingMultiOutputConfig {
    /// Mini-batch size for streaming updates
    pub batch_size: usize,
    /// Maximum buffer size before forced update
    pub max_buffer_size: usize,
    /// Learning rate
    pub learning_rate: Float,
    /// Whether to detect concept drift
    pub detect_drift: bool,
    /// Window size for drift detection
    pub drift_window_size: usize,
    /// Threshold for drift detection
    pub drift_threshold: Float,
}

impl Default for StreamingMultiOutputConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_buffer_size: 1000,
            learning_rate: 0.01,
            detect_drift: true,
            drift_window_size: 100,
            drift_threshold: 0.1,
        }
    }
}

/// Streaming Multi-Output Learner
///
/// Handles streaming data with mini-batch processing and concept drift detection.
///
/// # Examples
///
/// ```rust
/// use sklears_multioutput::streaming::{StreamingMultiOutput, StreamingMultiOutputConfig};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Predict};
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
///
/// let mut model = StreamingMultiOutput::new()
///     .batch_size(2)
///     .learning_rate(0.1);
///
/// let trained = model.fit(&X.view(), &y.view()).unwrap();
///
/// // Add streaming data
/// let X_stream = array![[4.0, 5.0]];
/// let y_stream = array![[4.0, 5.0]];
/// let updated = trained.update_stream(&X_stream.view(), &y_stream.view()).unwrap();
///
/// let predictions = updated.predict(&X.view()).unwrap();
/// assert_eq!(predictions.dim(), (3, 2));
/// ```
#[derive(Debug, Clone)]
pub struct StreamingMultiOutput<S = Untrained> {
    state: S,
    config: StreamingMultiOutputConfig,
}

/// Trained state for Streaming Multi-Output
#[derive(Debug, Clone)]
pub struct StreamingMultiOutputTrained {
    /// Base incremental model
    pub base_model: IncrementalMultiOutputRegressionTrained,
    /// Buffer for mini-batch processing
    pub buffer_X: VecDeque<Array1<Float>>,
    pub buffer_y: VecDeque<Array1<Float>>,
    /// Performance history for drift detection
    pub error_history: VecDeque<Float>,
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Number of drift events detected
    pub n_drift_events: usize,
    /// Configuration
    pub config: StreamingMultiOutputConfig,
}

impl StreamingMultiOutput<Untrained> {
    /// Create a new streaming multi-output learner
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: StreamingMultiOutputConfig::default(),
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: StreamingMultiOutputConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Enable/disable drift detection
    pub fn detect_drift(mut self, detect: bool) -> Self {
        self.config.detect_drift = detect;
        self
    }
}

impl Default for StreamingMultiOutput<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for StreamingMultiOutput<Untrained> {
    type Fitted = StreamingMultiOutput<StreamingMultiOutputTrained>;

    fn fit(self, X: &ArrayView2<Float>, y: &ArrayView2<Float>) -> SklResult<Self::Fitted> {
        // Initialize base model
        let base_config = IncrementalMultiOutputRegressionConfig {
            learning_rate: self.config.learning_rate,
            ..Default::default()
        };

        let base_model = IncrementalMultiOutputRegression::new()
            .config(base_config)
            .fit(X, y)?;

        Ok(StreamingMultiOutput {
            state: StreamingMultiOutputTrained {
                base_model: base_model.state,
                buffer_X: VecDeque::new(),
                buffer_y: VecDeque::new(),
                error_history: VecDeque::new(),
                drift_detected: false,
                n_drift_events: 0,
                config: self.config,
            },
            config: StreamingMultiOutputConfig::default(),
        })
    }
}

impl StreamingMultiOutput<StreamingMultiOutputTrained> {
    /// Update with streaming data
    pub fn update_stream(
        mut self,
        X: &ArrayView2<Float>,
        y: &ArrayView2<Float>,
    ) -> SklResult<Self> {
        // Add to buffer
        for i in 0..X.nrows() {
            self.state.buffer_X.push_back(X.row(i).to_owned());
            self.state.buffer_y.push_back(y.row(i).to_owned());
        }

        // Process if buffer is full
        if self.state.buffer_X.len() >= self.state.config.batch_size {
            self = self.process_buffer()?;
        }

        Ok(self)
    }

    /// Process the current buffer
    fn process_buffer(mut self) -> SklResult<Self> {
        let batch_size = self.state.config.batch_size.min(self.state.buffer_X.len());

        if batch_size == 0 {
            return Ok(self);
        }

        // Extract batch from buffer
        let mut X_batch = Array2::zeros((batch_size, self.state.base_model.n_features));
        let mut y_batch = Array2::zeros((batch_size, self.state.base_model.n_outputs));

        for i in 0..batch_size {
            let x = self.state.buffer_X.pop_front().unwrap();
            let y = self.state.buffer_y.pop_front().unwrap();
            X_batch.row_mut(i).assign(&x);
            y_batch.row_mut(i).assign(&y);
        }

        // Detect drift if enabled
        if self.state.config.detect_drift {
            let pred = self.predict(&X_batch.view())?;
            let error: Float = (&y_batch - &pred).mapv(|x| x.powi(2)).mean().unwrap();

            self.state.error_history.push_back(error);
            if self.state.error_history.len() > self.state.config.drift_window_size {
                self.state.error_history.pop_front();
            }

            // Check for drift
            if self.state.error_history.len() >= self.state.config.drift_window_size {
                let recent_error: Float = self
                    .state
                    .error_history
                    .iter()
                    .rev()
                    .take(self.state.config.drift_window_size / 2)
                    .sum::<Float>()
                    / (self.state.config.drift_window_size / 2) as Float;

                let old_error: Float = self
                    .state
                    .error_history
                    .iter()
                    .take(self.state.config.drift_window_size / 2)
                    .sum::<Float>()
                    / (self.state.config.drift_window_size / 2) as Float;

                if recent_error > old_error * (1.0 + self.state.config.drift_threshold) {
                    self.state.drift_detected = true;
                    self.state.n_drift_events += 1;
                    // Could reset model here if needed
                }
            }
        }

        // Update base model
        let base_wrapper = IncrementalMultiOutputRegression {
            state: self.state.base_model.clone(),
            config: IncrementalMultiOutputRegressionConfig::default(),
        };

        let updated = base_wrapper.partial_fit(&X_batch.view(), &y_batch.view())?;
        self.state.base_model = updated.state;

        Ok(self)
    }

    /// Force processing of remaining buffer
    pub fn flush_buffer(mut self) -> SklResult<Self> {
        while !self.state.buffer_X.is_empty() {
            self = self.process_buffer()?;
        }
        Ok(self)
    }

    /// Check if drift was detected
    pub fn drift_detected(&self) -> bool {
        self.state.drift_detected
    }

    /// Get number of drift events
    pub fn n_drift_events(&self) -> usize {
        self.state.n_drift_events
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.state.buffer_X.len()
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>>
    for StreamingMultiOutput<StreamingMultiOutputTrained>
{
    fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let base_wrapper = IncrementalMultiOutputRegression {
            state: self.state.base_model.clone(),
            config: IncrementalMultiOutputRegressionConfig::default(),
        };
        base_wrapper.predict(X)
    }
}

impl Estimator for StreamingMultiOutput<Untrained> {
    type Config = StreamingMultiOutputConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for StreamingMultiOutput<StreamingMultiOutputTrained> {
    type Config = StreamingMultiOutputConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.state.config
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_incremental_regression_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let model = IncrementalMultiOutputRegression::new()
            .learning_rate(0.1)
            .alpha(0.0001);

        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (3, 2));
        assert_eq!(trained.n_samples_seen(), 3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_incremental_regression_partial_fit() {
        let X1 = array![[1.0, 2.0], [2.0, 3.0]];
        let y1 = array![[1.0, 2.0], [2.0, 3.0]];

        let model = IncrementalMultiOutputRegression::new().learning_rate(0.1);
        let trained = model.fit(&X1.view(), &y1.view()).unwrap();

        // Partial fit with new data
        let X2 = array![[3.0, 4.0], [4.0, 5.0]];
        let y2 = array![[3.0, 4.0], [4.0, 5.0]];
        let updated = trained.partial_fit(&X2.view(), &y2.view()).unwrap();

        assert_eq!(updated.n_samples_seen(), 4);

        let predictions = updated.predict(&X2.view()).unwrap();
        assert_eq!(predictions.dim(), (2, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_incremental_regression_learning_rate_decay() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0]];

        let model = IncrementalMultiOutputRegression::new().learning_rate(0.1);
        let trained = model.fit(&X.view(), &y.view()).unwrap();

        let initial_lr = trained.current_learning_rate();

        // Partial fit should decay learning rate
        let X2 = array![[3.0, 4.0]];
        let y2 = array![[3.0, 4.0]];
        let updated = trained.partial_fit(&X2.view(), &y2.view()).unwrap();

        assert!(updated.current_learning_rate() < initial_lr);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let model = StreamingMultiOutput::new().batch_size(2).learning_rate(0.1);

        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (3, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_update() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0]];

        let model = StreamingMultiOutput::new().batch_size(2);
        let trained = model.fit(&X.view(), &y.view()).unwrap();

        // Stream new data
        let X_stream = array![[3.0, 4.0], [4.0, 5.0]];
        let y_stream = array![[3.0, 4.0], [4.0, 5.0]];
        let updated = trained
            .update_stream(&X_stream.view(), &y_stream.view())
            .unwrap();

        let predictions = updated.predict(&X_stream.view()).unwrap();
        assert_eq!(predictions.dim(), (2, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_buffer() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0]];

        let model = StreamingMultiOutput::new().batch_size(5); // Large batch size
        let trained = model.fit(&X.view(), &y.view()).unwrap();

        // Add small amount of data (should buffer)
        let X_stream = array![[3.0, 4.0]];
        let y_stream = array![[3.0, 4.0]];
        let updated = trained
            .update_stream(&X_stream.view(), &y_stream.view())
            .unwrap();

        assert_eq!(updated.buffer_size(), 1);

        // Flush buffer
        let flushed = updated.flush_buffer().unwrap();
        assert_eq!(flushed.buffer_size(), 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_drift_detection() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let model = StreamingMultiOutput::new()
            .batch_size(2)
            .detect_drift(true)
            .learning_rate(0.1);

        let trained = model.fit(&X.view(), &y.view()).unwrap();

        // The model should track drift events
        assert_eq!(trained.n_drift_events(), 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_incremental_regression_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]; // Mismatched

        let model = IncrementalMultiOutputRegression::new();
        assert!(model.fit(&X.view(), &y.view()).is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_incremental_regression_prediction_error() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0]];

        let model = IncrementalMultiOutputRegression::new();
        let trained = model.fit(&X.view(), &y.view()).unwrap();

        // Wrong number of features
        let X_test = array![[1.0]];
        assert!(trained.predict(&X_test.view()).is_err());
    }
}
