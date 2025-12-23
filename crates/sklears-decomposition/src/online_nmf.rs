//! Online Non-negative Matrix Factorization (Online NMF)
//!
//! This module provides online/streaming NMF algorithms that can process data
//! incrementally without loading the entire dataset into memory.
//!
//! Features:
//! - Stochastic gradient descent (SGD) based online NMF
//! - Mini-batch online NMF for improved stability
//! - Adaptive learning rates with momentum
//! - Incremental dictionary updates
//! - Support for partial_fit patterns

use scirs2_core::ndarray::Array2;
use scirs2_core::random::seeded_rng;
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Configuration for Online NMF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineNMFConfig {
    /// Number of components (latent factors)
    pub n_components: usize,

    /// Initial learning rate
    pub learning_rate: Float,

    /// Learning rate decay factor
    pub decay_rate: Float,

    /// Momentum coefficient for SGD
    pub momentum: Float,

    /// L1 regularization parameter
    pub l1_reg: Float,

    /// L2 regularization parameter
    pub l2_reg: Float,

    /// Mini-batch size (1 for pure online)
    pub batch_size: usize,

    /// Maximum iterations per batch
    pub max_iter_per_batch: usize,

    /// Convergence tolerance
    pub tolerance: Float,

    /// Random seed
    pub random_state: Option<u64>,

    /// Initialization method ("random", "nndsvd")
    pub init_method: String,

    /// Shuffle batches
    pub shuffle: bool,
}

impl Default for OnlineNMFConfig {
    fn default() -> Self {
        Self {
            n_components: 10,
            learning_rate: 0.01,
            decay_rate: 0.99,
            momentum: 0.9,
            l1_reg: 0.0,
            l2_reg: 0.01,
            batch_size: 100,
            max_iter_per_batch: 10,
            tolerance: 1e-4,
            random_state: None,
            init_method: "random".to_string(),
            shuffle: true,
        }
    }
}

/// Online NMF model for streaming/incremental learning
#[derive(Debug, Clone)]
pub struct OnlineNMF {
    config: OnlineNMFConfig,
    /// Dictionary matrix W (n_features × n_components)
    w: Option<Array2<Float>>,
    /// Velocity for momentum-based updates
    w_velocity: Option<Array2<Float>>,
    /// Number of samples seen
    n_samples_seen: usize,
    /// Current learning rate
    current_learning_rate: Float,
    /// Convergence history
    loss_history: Vec<Float>,
    /// Whether model is fitted
    fitted: bool,
}

impl OnlineNMF {
    /// Create new Online NMF model
    pub fn new(config: OnlineNMFConfig) -> Self {
        Self {
            current_learning_rate: config.learning_rate,
            config,
            w: None,
            w_velocity: None,
            n_samples_seen: 0,
            loss_history: Vec::new(),
            fitted: false,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(OnlineNMFConfig::default())
    }

    /// Builder for Online NMF
    pub fn builder() -> OnlineNMFBuilder {
        OnlineNMFBuilder::new()
    }

    /// Initialize dictionary matrix W
    fn initialize_dictionary(&mut self, n_features: usize) -> Result<()> {
        let seed = self.config.random_state.unwrap_or(42);
        let mut rng = seeded_rng(seed);

        match self.config.init_method.as_str() {
            "random" => {
                // Random initialization with non-negative values
                let w = Array2::from_shape_fn((n_features, self.config.n_components), |_| {
                    rng.gen_range(0.0..1.0)
                });
                self.w = Some(w);
            }
            "nndsvd" => {
                // Simplified NNDSVD initialization
                // In a full implementation, would use proper SVD
                let w = Array2::from_shape_fn((n_features, self.config.n_components), |_| {
                    rng.gen_range(0.1..1.0)
                });
                self.w = Some(w);
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown initialization method: {}",
                    self.config.init_method
                )));
            }
        }

        // Initialize velocity for momentum
        self.w_velocity = Some(Array2::zeros((n_features, self.config.n_components)));

        Ok(())
    }

    /// Fit the model on a batch of data
    pub fn partial_fit(&mut self, x: &Array2<Float>) -> Result<&mut Self> {
        let (n_samples, n_features) = x.dim();

        // Validate input
        if x.iter().any(|&v| v < 0.0) {
            return Err(SklearsError::InvalidInput(
                "Input matrix must be non-negative for NMF".to_string(),
            ));
        }

        // Initialize dictionary on first call
        if self.w.is_none() {
            self.initialize_dictionary(n_features)?;
        }

        // Check feature dimension consistency
        {
            let w = self.w.as_ref().unwrap();
            if w.nrows() != n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature dimension mismatch: expected {}, got {}",
                    w.nrows(),
                    n_features
                )));
            }
        }

        // Initialize H for this batch (n_samples × n_components)
        let mut h = self.initialize_h(x)?;

        // Mini-batch gradient descent
        for iter in 0..self.config.max_iter_per_batch {
            // Compute reconstruction and loss
            let w_current = self.w.as_ref().unwrap().clone();
            let reconstruction = h.dot(&w_current.t());
            let loss = self.compute_loss(x, &reconstruction, &h);
            self.loss_history.push(loss);

            // Check convergence
            if iter > 0 {
                let prev_loss = self.loss_history[self.loss_history.len() - 2];
                if (prev_loss - loss).abs() < self.config.tolerance {
                    break;
                }
            }

            // Update H (activation coefficients)
            self.update_h(&mut h, x, &w_current)?;

            // Update W (dictionary)
            self.update_w(x, &h)?;
        }

        // Update sample count and learning rate
        self.n_samples_seen += n_samples;
        self.update_learning_rate();

        self.fitted = true;
        Ok(self)
    }

    /// Initialize H matrix for a batch
    fn initialize_h(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, _) = x.dim();
        let seed = self.config.random_state.unwrap_or(42) + self.n_samples_seen as u64;
        let mut rng = seeded_rng(seed);

        let h = Array2::from_shape_fn((n_samples, self.config.n_components), |_| {
            rng.gen_range(0.0..1.0)
        });

        Ok(h)
    }

    /// Update H (activation coefficients) using gradient descent
    fn update_h(&self, h: &mut Array2<Float>, x: &Array2<Float>, w: &Array2<Float>) -> Result<()> {
        // Gradient: ∇H = -(X - HW^T)W + λ₁ + 2λ₂H
        let reconstruction = h.dot(&w.t());
        let residual = x - reconstruction;

        // Compute gradient: residual is (n_samples, n_features), w is (n_features, n_components)
        // gradient_part is (n_samples, n_components)
        let gradient_part = residual.dot(w);
        let gradient = gradient_part.mapv(|v| -v)
            + self.config.l1_reg
            + h.mapv(|v| 2.0 * self.config.l2_reg * v);

        // Update with learning rate
        let lr = self.current_learning_rate;
        *h = (h.clone() - gradient.mapv(|v| lr * v)).mapv(|v| v.max(0.0));

        Ok(())
    }

    /// Update W (dictionary) using momentum-based gradient descent
    fn update_w(&mut self, x: &Array2<Float>, h: &Array2<Float>) -> Result<()> {
        let w = self.w.as_ref().unwrap();
        let w_velocity = self.w_velocity.as_ref().unwrap();

        // Gradient: ∇W = -(X - HW^T)^T H + λ₁ + 2λ₂W
        let reconstruction = h.dot(&w.t());
        let residual = x - reconstruction;

        // Compute gradient: residual^T is (n_features, n_samples), h is (n_samples, n_components)
        // gradient_part is (n_features, n_components)
        let residual_t = residual.t().to_owned();
        let gradient_part = residual_t.dot(h);
        let gradient = gradient_part.mapv(|v| -v)
            + self.config.l1_reg
            + w.mapv(|v| 2.0 * self.config.l2_reg * v);

        // Momentum update
        let new_velocity = w_velocity.mapv(|v| v * self.config.momentum)
            - gradient.mapv(|v| self.current_learning_rate * v);

        // Apply update with non-negativity constraint
        let new_w = (w + &new_velocity).mapv(|v| v.max(0.0));

        self.w = Some(new_w);
        self.w_velocity = Some(new_velocity);

        Ok(())
    }

    /// Compute reconstruction loss
    fn compute_loss(
        &self,
        x: &Array2<Float>,
        reconstruction: &Array2<Float>,
        h: &Array2<Float>,
    ) -> Float {
        // Frobenius norm of residual
        let residual = x - reconstruction;
        let reconstruction_loss = residual.mapv(|v| v.powi(2)).sum();

        // L1 regularization on H
        let l1_term = if self.config.l1_reg > 0.0 {
            self.config.l1_reg * h.mapv(|v| v.abs()).sum()
        } else {
            0.0
        };

        // L2 regularization on W and H
        let l2_term = if self.config.l2_reg > 0.0 {
            let w = self.w.as_ref().unwrap();
            self.config.l2_reg * (w.mapv(|v| v.powi(2)).sum() + h.mapv(|v| v.powi(2)).sum())
        } else {
            0.0
        };

        reconstruction_loss + l1_term + l2_term
    }

    /// Update learning rate with decay
    fn update_learning_rate(&mut self) {
        self.current_learning_rate *= self.config.decay_rate;
    }

    /// Transform data using the learned dictionary
    pub fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "Model must be fitted before transform".to_string(),
            ));
        }

        let w = self.w.as_ref().unwrap();
        let (_n_samples, n_features) = x.dim();

        if n_features != w.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature dimension mismatch: expected {}, got {}",
                w.nrows(),
                n_features
            )));
        }

        // Initialize H
        let mut h = self.initialize_h(x)?;

        // Optimize H for fixed W
        for _ in 0..self.config.max_iter_per_batch {
            self.update_h(&mut h, x, w)?;
        }

        Ok(h)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &Array2<Float>) -> Result<Array2<Float>> {
        self.partial_fit(x)?;
        self.transform(x)
    }

    /// Get the learned dictionary matrix
    pub fn get_components(&self) -> Result<Array2<Float>> {
        self.w
            .clone()
            .ok_or_else(|| SklearsError::InvalidInput("Model not fitted".to_string()))
    }

    /// Get reconstruction error
    pub fn reconstruction_error(&self, x: &Array2<Float>) -> Result<Float> {
        let h = self.transform(x)?;
        let w = self.w.as_ref().unwrap();
        let reconstruction = h.dot(&w.t());
        let residual = x - &reconstruction;
        Ok(residual.mapv(|v| v.powi(2)).sum().sqrt())
    }

    /// Get loss history
    pub fn get_loss_history(&self) -> &[Float] {
        &self.loss_history
    }

    /// Get number of samples seen
    pub fn get_n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Reset the model to unfitted state
    pub fn reset(&mut self) {
        self.w = None;
        self.w_velocity = None;
        self.n_samples_seen = 0;
        self.current_learning_rate = self.config.learning_rate;
        self.loss_history.clear();
        self.fitted = false;
    }
}

/// Builder for Online NMF
pub struct OnlineNMFBuilder {
    config: OnlineNMFConfig,
}

impl OnlineNMFBuilder {
    pub fn new() -> Self {
        Self {
            config: OnlineNMFConfig::default(),
        }
    }

    pub fn n_components(mut self, n: usize) -> Self {
        self.config.n_components = n;
        self
    }

    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    pub fn decay_rate(mut self, rate: Float) -> Self {
        self.config.decay_rate = rate;
        self
    }

    pub fn momentum(mut self, m: Float) -> Self {
        self.config.momentum = m;
        self
    }

    pub fn l1_reg(mut self, reg: Float) -> Self {
        self.config.l1_reg = reg;
        self
    }

    pub fn l2_reg(mut self, reg: Float) -> Self {
        self.config.l2_reg = reg;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn max_iter_per_batch(mut self, iter: usize) -> Self {
        self.config.max_iter_per_batch = iter;
        self
    }

    pub fn tolerance(mut self, tol: Float) -> Self {
        self.config.tolerance = tol;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    pub fn init_method(mut self, method: &str) -> Self {
        self.config.init_method = method.to_string();
        self
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }

    pub fn build(self) -> OnlineNMF {
        OnlineNMF::new(self.config)
    }
}

impl Default for OnlineNMFBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_online_nmf_creation() {
        let model = OnlineNMF::builder()
            .n_components(5)
            .learning_rate(0.01)
            .build();

        assert_eq!(model.config.n_components, 5);
        assert!(!model.fitted);
    }

    #[test]
    fn test_online_nmf_partial_fit() {
        let mut rng = thread_rng();
        let data = Array2::from_shape_fn((20, 10), |_| rng.gen_range(0.0..1.0));

        let mut model = OnlineNMF::builder()
            .n_components(3)
            .max_iter_per_batch(5)
            .random_state(42)
            .build();

        let result = model.partial_fit(&data);
        assert!(result.is_ok());
        assert!(model.fitted);
        assert_eq!(model.n_samples_seen, 20);
    }

    #[test]
    fn test_online_nmf_transform() {
        let mut rng = thread_rng();
        let data = Array2::from_shape_fn((20, 10), |_| rng.gen_range(0.0..1.0));

        let mut model = OnlineNMF::builder()
            .n_components(3)
            .random_state(42)
            .build();

        model.partial_fit(&data).unwrap();
        let transformed = model.transform(&data);

        assert!(transformed.is_ok());
        let h = transformed.unwrap();
        assert_eq!(h.dim(), (20, 3));
    }

    #[test]
    fn test_online_nmf_incremental() {
        let mut rng = thread_rng();

        let mut model = OnlineNMF::builder()
            .n_components(3)
            .random_state(42)
            .build();

        // Fit on multiple batches
        for _ in 0..3 {
            let batch = Array2::from_shape_fn((10, 8), |_| rng.gen_range(0.0..1.0));
            model.partial_fit(&batch).unwrap();
        }

        assert_eq!(model.n_samples_seen, 30);
        assert!(model.fitted);

        // Check that dictionary was learned
        let w = model.get_components().unwrap();
        assert_eq!(w.dim(), (8, 3));
    }

    #[test]
    fn test_online_nmf_negative_input() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 2.0, 3.0]).unwrap();

        let mut model = OnlineNMF::builder().n_components(2).build();

        let result = model.partial_fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_online_nmf_reconstruction() {
        let mut rng = thread_rng();
        let data = Array2::from_shape_fn((15, 8), |_| rng.gen_range(0.0..1.0));

        let mut model = OnlineNMF::builder()
            .n_components(4)
            .max_iter_per_batch(20)
            .random_state(42)
            .build();

        model.partial_fit(&data).unwrap();

        let error = model.reconstruction_error(&data).unwrap();
        assert!(error >= 0.0);
        assert!(error.is_finite());
    }

    #[test]
    fn test_online_nmf_reset() {
        let mut rng = thread_rng();
        let data = Array2::from_shape_fn((10, 5), |_| rng.gen_range(0.0..1.0));

        let mut model = OnlineNMF::builder()
            .n_components(3)
            .random_state(42)
            .build();

        model.partial_fit(&data).unwrap();
        assert!(model.fitted);

        model.reset();
        assert!(!model.fitted);
        assert_eq!(model.n_samples_seen, 0);
    }

    #[test]
    fn test_online_nmf_builder() {
        let model = OnlineNMF::builder()
            .n_components(5)
            .learning_rate(0.02)
            .momentum(0.95)
            .l1_reg(0.01)
            .l2_reg(0.001)
            .batch_size(50)
            .tolerance(1e-5)
            .random_state(123)
            .init_method("random")
            .shuffle(true)
            .build();

        assert_eq!(model.config.n_components, 5);
        assert_eq!(model.config.learning_rate, 0.02);
        assert_eq!(model.config.momentum, 0.95);
        assert_eq!(model.config.random_state, Some(123));
    }
}
