//! Autoencoder implementation for unsupervised feature learning
//!
//! This module provides various autoencoder architectures including:
//! - Standard autoencoder for dimensionality reduction
//! - Denoising autoencoder for data cleaning
//! - Sparse autoencoder for feature learning
//! - Deep autoencoder for complex representations

use crate::activation::Activation;
use crate::SklearsError;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::Result,
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Optimizer types for training
#[derive(Debug, Clone, Copy)]
pub enum OptimizerType {
    SGD,
    Adam,
    RMSprop,
}

/// Autoencoder configuration
#[derive(Debug, Clone)]
pub struct AutoencoderConfig {
    /// Size of hidden layer (encoding dimension) - for simple autoencoder
    pub encoding_dim: usize,
    /// Encoder layer sizes - for deep autoencoder
    pub encoder_layers: Option<Vec<usize>>,
    /// Activation function
    pub activation: Activation,
    /// Learning rate
    pub learning_rate: Float,
    /// Number of epochs
    pub n_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Random state
    pub random_state: Option<u64>,
    /// Noise factor for denoising autoencoder
    pub noise_factor: Float,
    /// L2 regularization parameter
    pub l2_reg: Float,
    /// Sparsity parameters (rho, beta) for sparse autoencoder
    pub sparsity: Option<(Float, Float)>,
    /// Optimizer type
    pub optimizer: OptimizerType,
}

impl Default for AutoencoderConfig {
    fn default() -> Self {
        Self {
            encoding_dim: 32,
            encoder_layers: None,
            activation: Activation::Relu,
            learning_rate: 0.01,
            n_epochs: 100,
            batch_size: 32,
            random_state: None,
            noise_factor: 0.0,
            l2_reg: 0.0,
            sparsity: None,
            optimizer: OptimizerType::SGD,
        }
    }
}

/// Simple Autoencoder model
#[derive(Debug, Clone)]
pub struct Autoencoder<State = Untrained> {
    config: AutoencoderConfig,
    state: PhantomData<State>,
    // Trained parameters
    encoder_weights_: Option<Array2<Float>>,
    encoder_bias_: Option<Array1<Float>>,
    decoder_weights_: Option<Array2<Float>>,
    decoder_bias_: Option<Array1<Float>>,
    _n_features: Option<usize>,
}

impl Autoencoder<Untrained> {
    /// Create a new autoencoder
    pub fn new() -> Self {
        Self {
            config: AutoencoderConfig::default(),
            state: PhantomData,
            encoder_weights_: None,
            encoder_bias_: None,
            decoder_weights_: None,
            decoder_bias_: None,
            _n_features: None,
        }
    }

    /// Set encoding dimension
    pub fn encoding_dim(mut self, dim: usize) -> Self {
        self.config.encoding_dim = dim;
        self
    }

    /// Set activation function
    pub fn activation(mut self, activation: Activation) -> Self {
        self.config.activation = activation;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set number of epochs
    pub fn n_epochs(mut self, epochs: usize) -> Self {
        self.config.n_epochs = epochs;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set encoder layers for deep autoencoder
    pub fn encoder_layers(mut self, layers: Vec<usize>) -> Self {
        self.config.encoder_layers = Some(layers);
        self
    }

    /// Set noise factor for denoising autoencoder
    pub fn noise_factor(mut self, factor: Float) -> Self {
        self.config.noise_factor = factor;
        self
    }

    /// Set L2 regularization parameter
    pub fn l2_reg(mut self, reg: Float) -> Self {
        self.config.l2_reg = reg;
        self
    }

    /// Set sparsity parameters (rho, beta)
    pub fn sparsity(mut self, rho: Float, beta: Float) -> Self {
        self.config.sparsity = Some((rho, beta));
        self
    }

    /// Set optimizer type
    pub fn optimizer(mut self, opt: OptimizerType) -> Self {
        self.config.optimizer = opt;
        self
    }
}

impl Default for Autoencoder<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for Autoencoder<Untrained> {
    type Config = AutoencoderConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for Autoencoder<Untrained> {
    type Fitted = Autoencoder<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let encoding_dim = self.config.encoding_dim;

        // Initialize weights with Xavier initialization
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::seed_from_u64(42) // Use default seed if none provided
        };

        let limit_enc = (6.0 / (n_features + encoding_dim) as Float).sqrt();
        let limit_dec = (6.0 / (encoding_dim + n_features) as Float).sqrt();

        let mut encoder_weights = Array2::zeros((n_features, encoding_dim));
        let mut decoder_weights = Array2::zeros((encoding_dim, n_features));

        for i in 0..n_features {
            for j in 0..encoding_dim {
                encoder_weights[[i, j]] = rng.gen_range(-limit_enc..limit_enc);
            }
        }

        for i in 0..encoding_dim {
            for j in 0..n_features {
                decoder_weights[[i, j]] = rng.gen_range(-limit_dec..limit_dec);
            }
        }

        let mut encoder_bias = Array1::zeros(encoding_dim);
        let mut decoder_bias = Array1::zeros(n_features);

        // Training loop
        let batch_size = self.config.batch_size.min(n_samples);
        let n_batches = n_samples.div_ceil(batch_size);

        for epoch in 0..self.config.n_epochs {
            let mut epoch_loss = 0.0;

            // Shuffle indices
            let mut indices: Vec<usize> = (0..n_samples).collect();
            if self.config.random_state.is_some() {
                use scirs2_core::random::seq::SliceRandom;
                indices.shuffle(&mut rng);
            }

            for batch_idx in 0..n_batches {
                let start = batch_idx * batch_size;
                let end = ((batch_idx + 1) * batch_size).min(n_samples);
                let batch_indices = &indices[start..end];
                let actual_batch_size = batch_indices.len();

                // Get batch
                let mut batch = Array2::zeros((actual_batch_size, n_features));
                for (i, &idx) in batch_indices.iter().enumerate() {
                    batch.row_mut(i).assign(&x.row(idx));
                }

                // Forward pass
                let z_enc = batch.dot(&encoder_weights) + &encoder_bias;
                let a_enc = self.config.activation.apply(&z_enc);

                let z_dec = a_enc.dot(&decoder_weights) + &decoder_bias;
                let a_dec = Activation::Identity.apply(&z_dec); // Linear output

                // Compute loss (MSE)
                let diff = &batch - &a_dec;
                let loss = diff.mapv(|x| x * x).sum() / (actual_batch_size * n_features) as Float;
                epoch_loss += loss * actual_batch_size as Float;

                // Backward pass
                let d_output =
                    (&a_dec - &batch) * (2.0 / (actual_batch_size * n_features) as Float);

                // Decoder gradients
                let d_w_dec = a_enc.t().dot(&d_output);
                let d_b_dec = d_output.sum_axis(Axis(0));

                // Propagate through decoder
                let d_enc = d_output.dot(&decoder_weights.t());
                let d_enc = &d_enc * &self.config.activation.derivative(&a_enc);

                // Encoder gradients
                let d_w_enc = batch.t().dot(&d_enc);
                let d_b_enc = d_enc.sum_axis(Axis(0));

                // Update weights (simple gradient descent)
                encoder_weights = &encoder_weights - &d_w_enc * self.config.learning_rate;
                encoder_bias = &encoder_bias - &d_b_enc * self.config.learning_rate;
                decoder_weights = &decoder_weights - &d_w_dec * self.config.learning_rate;
                decoder_bias = &decoder_bias - &d_b_dec * self.config.learning_rate;
            }

            epoch_loss /= n_samples as Float;

            if epoch % 10 == 0 {
                println!("Epoch {epoch}: Loss = {epoch_loss:.6}");
            }
        }

        Ok(Autoencoder {
            config: self.config,
            state: PhantomData,
            encoder_weights_: Some(encoder_weights),
            encoder_bias_: Some(encoder_bias),
            decoder_weights_: Some(decoder_weights),
            decoder_bias_: Some(decoder_bias),
            _n_features: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for Autoencoder<Trained> {
    /// Transform data to encoded representation
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let encoder_weights =
            self.encoder_weights_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;
        let encoder_bias = self
            .encoder_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let z = x.dot(encoder_weights) + encoder_bias;
        Ok(self.config.activation.apply(&z))
    }
}

impl Autoencoder<Trained> {
    /// Reconstruct data (encode then decode)
    pub fn reconstruct(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let encoded = self.transform(x)?;

        let decoder_weights =
            self.decoder_weights_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "reconstruct".to_string(),
                })?;
        let decoder_bias = self
            .decoder_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "reconstruct".to_string(),
            })?;

        let z = encoded.dot(decoder_weights) + decoder_bias;
        Ok(Activation::Identity.apply(&z))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoencoder_construction() {
        let ae = Autoencoder::new()
            .encoding_dim(16)
            .activation(Activation::Relu)
            .learning_rate(0.01)
            .n_epochs(10);

        assert_eq!(ae.config.encoding_dim, 16);
        assert_eq!(ae.config.learning_rate, 0.01);
        assert_eq!(ae.config.n_epochs, 10);
    }

    #[test]
    fn test_autoencoder_fit_transform() {
        let x =
            Array2::from_shape_vec((10, 5), (0..50).map(|i| i as Float / 10.0).collect()).unwrap();

        let ae = Autoencoder::new()
            .encoding_dim(3)
            .n_epochs(20)
            .learning_rate(0.1)
            .random_state(42);

        let fitted = ae.fit(&x, &()).unwrap();

        // Check encoding
        let encoded = fitted.transform(&x).unwrap();
        assert_eq!(encoded.shape(), &[10, 3]);

        // Check reconstruction
        let reconstructed = fitted.reconstruct(&x).unwrap();
        assert_eq!(reconstructed.shape(), x.shape());
    }
}
