//! Online learning algorithms for linear models
//!
//! This module provides online learning capabilities for processing streaming data including:
//! - Stochastic gradient descent variants
//! - Online coordinate descent for streaming data
//! - Adaptive learning rate schedules
//! - Mini-batch processing capabilities
//!
//! Online learning allows models to be updated incrementally as new data arrives,
//! making them suitable for streaming applications and large datasets that don't fit in memory.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

use crate::Penalty;

/// Learning rate schedule for online algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant(Float),
    /// Inverse scaling: learning_rate / (t^power)
    InverseScaling { initial_lr: Float, power: Float },
    /// Exponential decay: initial_lr * (decay_rate^t)
    ExponentialDecay {
        initial_lr: Float,
        decay_rate: Float,
    },
    /// Step decay: reduce learning rate by factor every step_size epochs
    StepDecay {
        initial_lr: Float,
        step_size: usize,
        gamma: Float,
    },
    /// Adaptive learning rate based on loss changes
    Adaptive {
        initial_lr: Float,
        patience: usize,
        factor: Float,
        min_lr: Float,
    },
}

impl Default for LearningRateSchedule {
    fn default() -> Self {
        Self::Constant(0.01)
    }
}

/// SGD variant types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SGDVariant {
    /// Standard stochastic gradient descent
    Standard,
    /// SGD with momentum
    Momentum { momentum: Float },
    /// Nesterov accelerated gradient
    Nesterov { momentum: Float },
    /// AdaGrad adaptive learning rate
    AdaGrad { epsilon: Float },
    /// RMSprop adaptive learning rate
    RMSprop { decay: Float, epsilon: Float },
    /// Adam optimizer
    Adam {
        beta1: Float,
        beta2: Float,
        epsilon: Float,
    },
}

impl Default for SGDVariant {
    fn default() -> Self {
        Self::Standard
    }
}

/// Mini-batch processing configuration
#[derive(Debug, Clone)]
pub struct MiniBatchConfig {
    /// Batch size for mini-batch processing
    pub batch_size: usize,
    /// Whether to shuffle data within mini-batches
    pub shuffle: bool,
    /// Buffer size for streaming data
    pub buffer_size: usize,
    /// Whether to drop last incomplete batch
    pub drop_last: bool,
}

impl Default for MiniBatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            buffer_size: 1000,
            drop_last: false,
        }
    }
}

/// Configuration for online learning algorithms
#[derive(Debug, Clone)]
pub struct OnlineLearningConfig {
    /// Learning rate schedule
    pub learning_rate: LearningRateSchedule,
    /// SGD variant to use
    pub sgd_variant: SGDVariant,
    /// Regularization penalty
    pub penalty: Penalty,
    /// Maximum number of iterations/epochs
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Mini-batch configuration
    pub mini_batch: MiniBatchConfig,
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: LearningRateSchedule::default(),
            sgd_variant: SGDVariant::default(),
            penalty: Penalty::None,
            max_iter: 1000,
            tol: 1e-4,
            random_state: None,
            fit_intercept: true,
            mini_batch: MiniBatchConfig::default(),
        }
    }
}

/// Online linear regression using stochastic gradient descent
#[derive(Debug, Clone)]
pub struct OnlineLinearRegression {
    config: OnlineLearningConfig,
    // Model parameters
    weights: Option<Array1<Float>>,
    intercept: Option<Float>,
    n_features: Option<usize>,
    // SGD state
    iteration: usize,
    momentum_buffer: Option<Array1<Float>>,
    squared_gradients: Option<Array1<Float>>, // For AdaGrad/RMSprop
    adam_m: Option<Array1<Float>>,            // Adam first moment
    adam_v: Option<Array1<Float>>,            // Adam second moment
    // Loss tracking
    loss_history: Vec<Float>,
    adaptive_lr_state: AdaptiveLRState,
}

/// State for adaptive learning rate scheduling
#[derive(Debug, Clone)]
struct AdaptiveLRState {
    best_loss: Float,
    patience_counter: usize,
    current_lr: Float,
}

impl Default for AdaptiveLRState {
    fn default() -> Self {
        Self {
            best_loss: Float::INFINITY,
            patience_counter: 0,
            current_lr: 0.01,
        }
    }
}

/// Online logistic regression using stochastic gradient descent
#[derive(Debug, Clone)]
pub struct OnlineLogisticRegression {
    config: OnlineLearningConfig,
    // Model parameters
    weights: Option<Array1<Float>>,
    intercept: Option<Float>,
    n_features: Option<usize>,
    classes: Option<Array1<Float>>,
    // SGD state
    iteration: usize,
    momentum_buffer: Option<Array1<Float>>,
    squared_gradients: Option<Array1<Float>>,
    adam_m: Option<Array1<Float>>,
    adam_v: Option<Array1<Float>>,
    // Loss tracking
    loss_history: Vec<Float>,
    adaptive_lr_state: AdaptiveLRState,
}

/// Online coordinate descent for sparse regression
#[derive(Debug, Clone)]
pub struct OnlineCoordinateDescent {
    config: OnlineLearningConfig,
    weights: Option<Array1<Float>>,
    intercept: Option<Float>,
    n_features: Option<usize>,
    // Coordinate descent state
    coordinate_order: Vec<usize>,
    feature_variances: Option<Array1<Float>>,
    loss_history: Vec<Float>,
}

/// Mini-batch iterator for streaming data processing
#[derive(Debug)]
pub struct MiniBatchIterator<'a> {
    x: ArrayView2<'a, Float>,
    y: ArrayView1<'a, Float>,
    config: MiniBatchConfig,
    current_idx: usize,
    indices: Vec<usize>,
}

impl OnlineLinearRegression {
    /// Create a new online linear regression model
    pub fn new() -> Self {
        Self {
            config: OnlineLearningConfig::default(),
            weights: None,
            intercept: None,
            n_features: None,
            iteration: 0,
            momentum_buffer: None,
            squared_gradients: None,
            adam_m: None,
            adam_v: None,
            loss_history: Vec::new(),
            adaptive_lr_state: AdaptiveLRState::default(),
        }
    }

    /// Configure learning rate schedule
    pub fn learning_rate(mut self, schedule: LearningRateSchedule) -> Self {
        self.config.learning_rate = schedule;
        if let LearningRateSchedule::Adaptive { initial_lr, .. } = schedule {
            self.adaptive_lr_state.current_lr = initial_lr;
        }
        self
    }

    /// Set SGD variant
    pub fn sgd_variant(mut self, variant: SGDVariant) -> Self {
        self.config.sgd_variant = variant;
        self
    }

    /// Set penalty for regularization
    pub fn penalty(mut self, penalty: Penalty) -> Self {
        self.config.penalty = penalty;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set mini-batch configuration
    pub fn mini_batch_config(mut self, config: MiniBatchConfig) -> Self {
        self.config.mini_batch = config;
        self
    }

    /// Fit the model using online learning
    pub fn fit(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Initialize model parameters
        if self.weights.is_none() {
            self.weights = Some(Array1::zeros(n_features));
            self.n_features = Some(n_features);

            if self.config.fit_intercept {
                self.intercept = Some(0.0);
            }

            // Initialize SGD state based on variant
            match self.config.sgd_variant {
                SGDVariant::Momentum { .. } | SGDVariant::Nesterov { .. } => {
                    self.momentum_buffer = Some(Array1::zeros(n_features));
                }
                SGDVariant::AdaGrad { .. } | SGDVariant::RMSprop { .. } => {
                    self.squared_gradients = Some(Array1::zeros(n_features));
                }
                SGDVariant::Adam { .. } => {
                    self.adam_m = Some(Array1::zeros(n_features));
                    self.adam_v = Some(Array1::zeros(n_features));
                }
                _ => {}
            }
        }

        // Process data in mini-batches
        let mut batch_iterator =
            MiniBatchIterator::new(x.view(), y.view(), &self.config.mini_batch);

        for epoch in 0..self.config.max_iter {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            batch_iterator.reset();
            while let Some((batch_x, batch_y)) = batch_iterator.next() {
                let loss = self.update_weights(&batch_x, &batch_y)?;
                epoch_loss += loss;
                n_batches += 1;
                self.iteration += 1;
            }

            epoch_loss /= n_batches as Float;
            self.loss_history.push(epoch_loss);

            // Update learning rate
            self.update_learning_rate(epoch_loss);

            // Check for convergence
            if self.loss_history.len() > 1 {
                let prev_loss = self.loss_history[self.loss_history.len() - 2];
                if (prev_loss - epoch_loss).abs() < self.config.tol {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Partial fit for online learning with new data
    pub fn partial_fit(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Initialize if first call
        if self.weights.is_none() {
            self.weights = Some(Array1::zeros(n_features));
            self.n_features = Some(n_features);

            if self.config.fit_intercept {
                self.intercept = Some(0.0);
            }
        }

        // Process single batch
        let loss = self.update_weights(x, y)?;
        self.loss_history.push(loss);
        self.iteration += 1;

        Ok(())
    }

    /// Update model weights using SGD
    fn update_weights(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Float> {
        // Get current learning rate before borrowing weights mutably
        let lr = self.get_current_learning_rate();
        let weights = self.weights.as_mut().unwrap();
        let n_samples = x.nrows();

        // Compute predictions
        let mut predictions = x.dot(weights);
        if let Some(intercept) = self.intercept {
            predictions += intercept;
        }

        // Compute loss and gradients
        let residuals = &predictions - y;
        let loss = residuals.mapv(|r| r * r).sum() / (2.0 * n_samples as Float);

        // Compute gradient
        let mut gradient = x.t().dot(&residuals) / n_samples as Float;

        // Add regularization gradient
        match self.config.penalty {
            Penalty::L2(alpha) => {
                gradient += &(alpha * &*weights);
            }
            Penalty::L1(alpha) => {
                gradient += &(alpha * weights.mapv(|w| w.signum()));
            }
            Penalty::ElasticNet { l1_ratio, alpha } => {
                let l1_term = alpha * l1_ratio * &weights.mapv(|w| w.signum());
                let l2_term = alpha * (1.0 - l1_ratio) * &*weights;
                gradient += &(&l1_term + &l2_term);
            }
            _ => {}
        }

        // Learning rate already obtained above

        // Apply SGD variant
        match self.config.sgd_variant {
            SGDVariant::Standard => {
                *weights -= &(lr * &gradient);
            }
            SGDVariant::Momentum { momentum } => {
                let momentum_buffer = self.momentum_buffer.as_mut().unwrap();
                *momentum_buffer = momentum * &*momentum_buffer + &gradient;
                *weights -= &(lr * &*momentum_buffer);
            }
            SGDVariant::Nesterov { momentum } => {
                let momentum_buffer = self.momentum_buffer.as_mut().unwrap();
                let prev_momentum = momentum_buffer.clone();
                *momentum_buffer = momentum * &*momentum_buffer + &gradient;
                *weights -= &(lr * (momentum * &*momentum_buffer + &prev_momentum));
            }
            SGDVariant::AdaGrad { epsilon } => {
                let squared_gradients = self.squared_gradients.as_mut().unwrap();
                *squared_gradients += &gradient.mapv(|g| g * g);
                let adapted_lr = &squared_gradients.mapv(|sg| lr / (sg.sqrt() + epsilon));
                *weights -= &(&gradient * adapted_lr);
            }
            SGDVariant::RMSprop { decay, epsilon } => {
                let squared_gradients = self.squared_gradients.as_mut().unwrap();
                *squared_gradients =
                    decay * &*squared_gradients + (1.0 - decay) * &gradient.mapv(|g| g * g);
                let adapted_lr = &squared_gradients.mapv(|sg| lr / (sg.sqrt() + epsilon));
                *weights -= &(&gradient * adapted_lr);
            }
            SGDVariant::Adam {
                beta1,
                beta2,
                epsilon,
            } => {
                let adam_m = self.adam_m.as_mut().unwrap();
                let adam_v = self.adam_v.as_mut().unwrap();

                *adam_m = beta1 * &*adam_m + (1.0 - beta1) * &gradient;
                *adam_v = beta2 * &*adam_v + (1.0 - beta2) * &gradient.mapv(|g| g * g);

                // Bias correction
                let t = self.iteration as Float + 1.0;
                let m_hat = &*adam_m / (1.0 - beta1.powf(t));
                let v_hat = &*adam_v / (1.0 - beta2.powf(t));

                let adapted_gradient = &m_hat / &(v_hat.mapv(|v| v.sqrt()) + epsilon);
                *weights -= &(lr * &adapted_gradient);
            }
        }

        // Update intercept if needed
        if self.config.fit_intercept {
            let intercept_gradient = residuals.sum() / n_samples as Float;
            let intercept = self.intercept.as_mut().unwrap();
            *intercept -= lr * intercept_gradient;
        }

        Ok(loss)
    }

    /// Get current learning rate based on schedule
    fn get_current_learning_rate(&self) -> Float {
        match self.config.learning_rate {
            LearningRateSchedule::Constant(lr) => lr,
            LearningRateSchedule::InverseScaling { initial_lr, power } => {
                initial_lr / (self.iteration as Float + 1.0).powf(power)
            }
            LearningRateSchedule::ExponentialDecay {
                initial_lr,
                decay_rate,
            } => initial_lr * decay_rate.powf(self.iteration as Float),
            LearningRateSchedule::StepDecay {
                initial_lr,
                step_size,
                gamma,
            } => {
                let steps = self.iteration / step_size;
                initial_lr * gamma.powf(steps as Float)
            }
            LearningRateSchedule::Adaptive { .. } => self.adaptive_lr_state.current_lr,
        }
    }

    /// Update learning rate for adaptive schedules
    fn update_learning_rate(&mut self, current_loss: Float) {
        if let LearningRateSchedule::Adaptive {
            patience,
            factor,
            min_lr,
            ..
        } = self.config.learning_rate
        {
            if current_loss < self.adaptive_lr_state.best_loss {
                self.adaptive_lr_state.best_loss = current_loss;
                self.adaptive_lr_state.patience_counter = 0;
            } else {
                self.adaptive_lr_state.patience_counter += 1;
                if self.adaptive_lr_state.patience_counter >= patience {
                    self.adaptive_lr_state.current_lr *= factor;
                    self.adaptive_lr_state.current_lr =
                        self.adaptive_lr_state.current_lr.max(min_lr);
                    self.adaptive_lr_state.patience_counter = 0;
                }
            }
        }
    }

    /// Predict using the trained model
    pub fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = x.dot(weights);
        if let Some(intercept) = self.intercept {
            predictions += intercept;
        }

        Ok(predictions)
    }

    /// Get model parameters
    pub fn weights(&self) -> Option<&Array1<Float>> {
        self.weights.as_ref()
    }

    /// Get intercept
    pub fn intercept(&self) -> Option<Float> {
        self.intercept
    }

    /// Get loss history
    pub fn loss_history(&self) -> &[Float] {
        &self.loss_history
    }
}

impl<'a> MiniBatchIterator<'a> {
    /// Create a new mini-batch iterator
    pub fn new(
        x: ArrayView2<'a, Float>,
        y: ArrayView1<'a, Float>,
        config: &MiniBatchConfig,
    ) -> Self {
        let n_samples = x.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if config.shuffle {
            // Simple shuffle (deterministic for reproducibility)
            for i in (1..indices.len()).rev() {
                let j = (i * 37) % (i + 1); // Simple deterministic "shuffle"
                indices.swap(i, j);
            }
        }

        Self {
            x,
            y,
            config: config.clone(),
            current_idx: 0,
            indices,
        }
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.config.shuffle {
            // Re-shuffle
            for i in (1..self.indices.len()).rev() {
                let j = (i * 41) % (i + 1); // Different seed for variety
                self.indices.swap(i, j);
            }
        }
    }

    /// Get next mini-batch
    pub fn next(&mut self) -> Option<(Array2<Float>, Array1<Float>)> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.config.batch_size).min(self.indices.len());

        if self.config.drop_last && (end_idx - self.current_idx) < self.config.batch_size {
            return None;
        }

        let batch_indices = &self.indices[self.current_idx..end_idx];

        // Extract batch data
        let mut batch_x = Array2::zeros((batch_indices.len(), self.x.ncols()));
        let mut batch_y = Array1::zeros(batch_indices.len());

        for (i, &idx) in batch_indices.iter().enumerate() {
            batch_x.row_mut(i).assign(&self.x.row(idx));
            batch_y[i] = self.y[idx];
        }

        self.current_idx = end_idx;
        Some((batch_x, batch_y))
    }
}

impl Default for OnlineLinearRegression {
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
    fn test_online_linear_regression_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![3.0, 5.0, 7.0, 9.0]; // y = x1 + x2

        let mut model = OnlineLinearRegression::new()
            .learning_rate(LearningRateSchedule::Constant(0.01))
            .max_iter(1000);

        model.fit(&x, &y).unwrap();

        // Test predictions
        let test_x = array![[5.0, 6.0]];
        let predictions = model.predict(&test_x).unwrap();

        // Should predict close to 11.0
        assert!((predictions[0] - 11.0).abs() < 1.0);
    }

    #[test]
    fn test_sgd_variants() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];
        let y = array![2.0, 4.0, 6.0];

        // Test different SGD variants
        let variants = vec![
            SGDVariant::Standard,
            SGDVariant::Momentum { momentum: 0.9 },
            SGDVariant::AdaGrad { epsilon: 1e-8 },
            SGDVariant::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        ];

        for variant in variants {
            let mut model = OnlineLinearRegression::new()
                .sgd_variant(variant)
                .learning_rate(LearningRateSchedule::Constant(0.01))
                .max_iter(100);

            assert!(model.fit(&x, &y).is_ok());
            assert!(model.weights().is_some());
        }
    }

    #[test]
    fn test_learning_rate_schedules() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];

        let schedules = vec![
            LearningRateSchedule::Constant(0.01),
            LearningRateSchedule::InverseScaling {
                initial_lr: 0.1,
                power: 0.5,
            },
            LearningRateSchedule::ExponentialDecay {
                initial_lr: 0.1,
                decay_rate: 0.99,
            },
        ];

        for schedule in schedules {
            let mut model = OnlineLinearRegression::new()
                .learning_rate(schedule)
                .max_iter(50);

            assert!(model.fit(&x, &y).is_ok());
        }
    }

    #[test]
    fn test_partial_fit() {
        let mut model =
            OnlineLinearRegression::new().learning_rate(LearningRateSchedule::Constant(0.01));

        // Fit data incrementally
        let x1 = array![[1.0, 2.0]];
        let y1 = array![3.0];
        model.partial_fit(&x1, &y1).unwrap();

        let x2 = array![[2.0, 3.0]];
        let y2 = array![5.0];
        model.partial_fit(&x2, &y2).unwrap();

        assert!(model.weights().is_some());
        assert_eq!(model.loss_history().len(), 2);
    }

    #[test]
    fn test_mini_batch_iterator() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = MiniBatchConfig {
            batch_size: 2,
            shuffle: false,
            buffer_size: 100,
            drop_last: false,
        };

        let mut iterator = MiniBatchIterator::new(x.view(), y.view(), &config);

        let mut batch_count = 0;
        while let Some((batch_x, batch_y)) = iterator.next() {
            batch_count += 1;
            assert!(batch_x.nrows() <= 2);
            assert_eq!(batch_x.nrows(), batch_y.len());
        }

        assert_eq!(batch_count, 3); // 2, 2, 1 samples per batch
    }
}
