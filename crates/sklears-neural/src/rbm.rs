//! Restricted Boltzmann Machine (RBM) implementation
//!
//! RBMs are generative stochastic artificial neural networks that can learn
//! a probability distribution over its set of inputs.

use crate::SklearsError;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::random::{seeded_rng, Distribution, RandomExt, Rng};
use sklears_core::{
    error::Result,
    traits::{Estimator, Fit, Predict, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Restricted Boltzmann Machine configuration
#[derive(Debug, Clone)]
pub struct RBMConfig {
    /// Number of hidden units
    pub n_hidden: usize,
    /// Learning rate
    pub learning_rate: Float,
    /// Batch size for mini-batch learning
    pub batch_size: usize,
    /// Number of epochs
    pub n_epochs: usize,
    /// Number of Gibbs sampling steps (for CD-k)
    pub n_gibbs: usize,
    /// L2 regularization coefficient
    pub l2_reg: Float,
    /// Momentum for weight updates
    pub momentum: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for RBMConfig {
    fn default() -> Self {
        Self {
            n_hidden: 256,
            learning_rate: 0.01,
            batch_size: 32,
            n_epochs: 10,
            n_gibbs: 1,
            l2_reg: 0.0,
            momentum: 0.0,
            random_state: None,
        }
    }
}

/// Restricted Boltzmann Machine
#[derive(Debug, Clone)]
pub struct RBM<State = Untrained> {
    config: RBMConfig,
    state: PhantomData<State>,
    // Trained parameters
    weights_: Option<Array2<Float>>,
    hidden_bias_: Option<Array1<Float>>,
    visible_bias_: Option<Array1<Float>>,
    n_features_: Option<usize>,
}

impl RBM<Untrained> {
    /// Create a new RBM
    pub fn new() -> Self {
        Self {
            config: RBMConfig::default(),
            state: PhantomData,
            weights_: None,
            hidden_bias_: None,
            visible_bias_: None,
            n_features_: None,
        }
    }

    /// Set the number of hidden units
    pub fn n_hidden(mut self, n_hidden: usize) -> Self {
        self.config.n_hidden = n_hidden;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set the number of epochs
    pub fn n_epochs(mut self, n_epochs: usize) -> Self {
        self.config.n_epochs = n_epochs;
        self
    }

    /// Set the number of Gibbs sampling steps
    pub fn n_gibbs(mut self, n_gibbs: usize) -> Self {
        self.config.n_gibbs = n_gibbs;
        self
    }

    /// Set L2 regularization
    pub fn l2_reg(mut self, l2_reg: Float) -> Self {
        self.config.l2_reg = l2_reg;
        self
    }

    /// Set momentum
    pub fn momentum(mut self, momentum: Float) -> Self {
        self.config.momentum = momentum;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }
}

impl Default for RBM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RBM<Untrained> {
    type Config = RBMConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for RBM<Untrained> {
    type Fitted = RBM<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_hidden = self.config.n_hidden;

        // Initialize weights and biases
        let mut rng = seeded_rng(42);
        let normal = Normal::new(0.0, 0.01).unwrap();
        let mut weights =
            Array2::from_shape_fn((n_features, n_hidden), |_| normal.sample(&mut rng));
        let mut hidden_bias = Array1::zeros(n_hidden);
        let mut visible_bias = Array1::zeros(n_features);

        // Velocity for momentum
        let mut weights_velocity = Array2::zeros((n_features, n_hidden));
        let mut hidden_bias_velocity = Array1::zeros(n_hidden);
        let mut visible_bias_velocity = Array1::zeros(n_features);

        // Training loop
        let batch_size = self.config.batch_size.min(n_samples);
        let n_batches = n_samples.div_ceil(batch_size);

        for epoch in 0..self.config.n_epochs {
            let mut reconstruction_error = 0.0;

            // Shuffle data
            let indices: Vec<usize> = if let Some(seed) = self.config.random_state {
                let mut rng = seeded_rng(seed + epoch as u64);
                let mut indices: Vec<usize> = (0..n_samples).collect();
                // Manual Fisher-Yates shuffle
                for i in (1..indices.len()).rev() {
                    let j = rng.gen_range(0..=i);
                    indices.swap(i, j);
                }
                indices
            } else {
                let mut indices: Vec<usize> = (0..n_samples).collect();
                // Manual Fisher-Yates shuffle
                for i in (1..indices.len()).rev() {
                    let j = rng.gen_range(0..=i);
                    indices.swap(i, j);
                }
                indices
            };

            // Mini-batch training
            for batch_idx in 0..n_batches {
                let start = batch_idx * batch_size;
                let end = ((batch_idx + 1) * batch_size).min(n_samples);
                let batch_indices = &indices[start..end];
                let actual_batch_size = batch_indices.len();

                // Get batch data
                let mut batch = Array2::zeros((actual_batch_size, n_features));
                for (i, &idx) in batch_indices.iter().enumerate() {
                    batch.row_mut(i).assign(&x.row(idx));
                }

                // Positive phase
                let (h_prob_pos, h_sample_pos) = sample_hidden(&batch, &weights, &hidden_bias);

                // Negative phase (CD-k)
                let mut v_sample = batch.clone();
                let mut h_sample = h_sample_pos.clone();

                for _ in 0..self.config.n_gibbs {
                    let (_v_prob, v_samp) = sample_visible(&h_sample, &weights, &visible_bias);
                    v_sample = v_samp;
                    let (_h_prob, h_samp) = sample_hidden(&v_sample, &weights, &hidden_bias);
                    h_sample = h_samp;
                }

                // Compute gradients
                let positive_grad = batch.t().dot(&h_prob_pos) / actual_batch_size as Float;
                let negative_grad = v_sample.t().dot(&h_sample) / actual_batch_size as Float;
                let weight_grad = &positive_grad - &negative_grad;

                let h_bias_grad = (h_prob_pos.sum_axis(Axis(0)) - h_sample.sum_axis(Axis(0)))
                    / actual_batch_size as Float;
                let v_bias_grad = (batch.sum_axis(Axis(0)) - v_sample.sum_axis(Axis(0)))
                    / actual_batch_size as Float;

                // Apply L2 regularization
                let weight_grad = if self.config.l2_reg > 0.0 {
                    weight_grad - self.config.l2_reg * &weights
                } else {
                    weight_grad
                };

                // Update with momentum
                weights_velocity = self.config.momentum * weights_velocity
                    + self.config.learning_rate * weight_grad;
                hidden_bias_velocity = self.config.momentum * hidden_bias_velocity
                    + self.config.learning_rate * h_bias_grad;
                visible_bias_velocity = self.config.momentum * visible_bias_velocity
                    + self.config.learning_rate * v_bias_grad;

                weights += &weights_velocity;
                hidden_bias += &hidden_bias_velocity;
                visible_bias += &visible_bias_velocity;

                // Compute reconstruction error
                let error = (&batch - &v_sample).mapv(|x| x.powi(2)).sum();
                reconstruction_error += error;
            }

            // Log progress
            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: Reconstruction error = {:.6}",
                    epoch,
                    reconstruction_error / n_samples as Float
                );
            }
        }

        Ok(RBM {
            config: self.config,
            state: PhantomData,
            weights_: Some(weights),
            hidden_bias_: Some(hidden_bias),
            visible_bias_: Some(visible_bias),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for RBM<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;
        let hidden_bias = self
            .hidden_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        // Compute hidden layer probabilities
        let hidden_probs = sigmoid(&(x.dot(weights) + hidden_bias));
        Ok(hidden_probs)
    }
}

impl RBM<Trained> {
    /// Reconstruct visible layer from data
    pub fn reconstruct(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "reconstruct".to_string(),
            })?;
        let hidden_bias = self
            .hidden_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "reconstruct".to_string(),
            })?;
        let visible_bias = self
            .visible_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "reconstruct".to_string(),
            })?;

        // Forward pass
        let (_, h_sample) = sample_hidden(x, weights, hidden_bias);

        // Backward pass
        let (v_prob, _) = sample_visible(&h_sample, weights, visible_bias);

        Ok(v_prob)
    }

    /// Generate new samples from the model
    pub fn sample(&self, n_samples: usize, n_gibbs_steps: usize) -> Result<Array2<Float>> {
        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "sample".to_string(),
            })?;
        let hidden_bias = self
            .hidden_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "sample".to_string(),
            })?;
        let visible_bias = self
            .visible_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "sample".to_string(),
            })?;
        let n_features = self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "sample".to_string(),
        })?;

        // Start from random visible state
        let mut rng = seeded_rng(42);
        let uniform = Uniform::new(0.0, 1.0).unwrap();
        let mut v_sample =
            Array2::from_shape_fn((n_samples, n_features), |_| uniform.sample(&mut rng));

        // Run Gibbs sampling
        for _ in 0..n_gibbs_steps {
            let (_, h_sample) = sample_hidden(&v_sample, weights, hidden_bias);
            let (_v_prob, v_samp) = sample_visible(&h_sample, weights, visible_bias);
            v_sample = v_samp;
        }

        Ok(v_sample)
    }

    /// Get the learned weights
    pub fn weights(&self) -> &Array2<Float> {
        self.weights_.as_ref().unwrap()
    }

    /// Get the hidden bias
    pub fn hidden_bias(&self) -> &Array1<Float> {
        self.hidden_bias_.as_ref().unwrap()
    }

    /// Get the visible bias
    pub fn visible_bias(&self) -> &Array1<Float> {
        self.visible_bias_.as_ref().unwrap()
    }
}

// Helper functions

fn sigmoid(x: &Array2<Float>) -> Array2<Float> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn sample_hidden(
    v: &Array2<Float>,
    w: &Array2<Float>,
    b_h: &Array1<Float>,
) -> (Array2<Float>, Array2<Float>) {
    let h_prob = sigmoid(&(v.dot(w) + b_h));
    let h_sample = sample_bernoulli(&h_prob);
    (h_prob, h_sample)
}

fn sample_visible(
    h: &Array2<Float>,
    w: &Array2<Float>,
    b_v: &Array1<Float>,
) -> (Array2<Float>, Array2<Float>) {
    let v_prob = sigmoid(&(h.dot(&w.t()) + b_v));
    let v_sample = sample_bernoulli(&v_prob);
    (v_prob, v_sample)
}

fn sample_bernoulli(probs: &Array2<Float>) -> Array2<Float> {
    let mut rng = seeded_rng(42);
    let shape = probs.shape();
    let mut samples = Array2::zeros((shape[0], shape[1]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if rng.gen::<Float>() < probs[[i, j]] {
                samples[[i, j]] = 1.0;
            }
        }
    }

    samples
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_rbm_construction() {
        let rbm = RBM::new().n_hidden(128).learning_rate(0.01).n_epochs(5);

        assert_eq!(rbm.config.n_hidden, 128);
        assert_eq!(rbm.config.learning_rate, 0.01);
        assert_eq!(rbm.config.n_epochs, 5);
    }

    #[test]
    fn test_rbm_fit_transform() {
        // Simple binary data
        let x = array![
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ];

        let rbm = RBM::new()
            .n_hidden(2)
            .n_epochs(10)
            .learning_rate(0.1)
            .random_state(42);

        let fitted = rbm.fit(&x, &()).unwrap();

        // Check that transform works
        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed.shape(), &[4, 2]);

        // Check that values are probabilities
        for val in transformed.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_rbm_reconstruction() {
        let x = array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],];

        let rbm = RBM::new()
            .n_hidden(3)
            .n_epochs(50)
            .learning_rate(0.1)
            .random_state(42);

        let fitted = rbm.fit(&x, &()).unwrap();
        let reconstructed = fitted.reconstruct(&x).unwrap();

        assert_eq!(reconstructed.shape(), x.shape());

        // Check that reconstructed values are probabilities
        for val in reconstructed.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_rbm_sampling() {
        let x = array![
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ];

        let rbm = RBM::new().n_hidden(2).n_epochs(20).random_state(42);

        let fitted = rbm.fit(&x, &()).unwrap();
        let samples = fitted.sample(5, 10).unwrap();

        assert_eq!(samples.shape(), &[5, 4]);

        // Check that samples are binary
        for val in samples.iter() {
            assert!(*val == 0.0 || *val == 1.0);
        }
    }
}
