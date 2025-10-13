//! Generative Adversarial Networks (GANs) implementation
//!
//! This module provides basic GAN implementations using simple fully connected networks.
//! GANs consist of two neural networks competing against each other: a generator that
//! creates fake data and a discriminator that tries to distinguish real from fake data.
//!
//! # Theory
//!
//! The GAN objective is a minimax game:
//! min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
//!
//! Where:
//! - G is the generator network
//! - D is the discriminator network
//! - x is real data
//! - z is noise input
//!
//! # Example
//!
//! ```rust
//! use sklears_neural::gan::{SimpleGAN, GANConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! let config = GANConfig::default()
//!     .latent_dim(100)
//!     .learning_rate(0.0002);
//!
//! let mut gan = SimpleGAN::new(config, 784, 28); // MNIST-like dimensions
//! ```

use crate::activation::Activation;
use crate::{NeuralResult, SklearsError};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::random::{thread_rng, Distribution, Rng};
use sklears_core::types::FloatBounds;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// GAN configuration parameters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GANConfig {
    /// Dimension of latent noise vector
    pub latent_dim: usize,
    /// Generator hidden layer sizes
    pub generator_layers: Vec<usize>,
    /// Discriminator hidden layer sizes
    pub discriminator_layers: Vec<usize>,
    /// Learning rate for both networks
    pub learning_rate: f64,
    /// Number of training epochs
    pub n_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Random seed
    pub random_state: Option<u64>,
}

impl Default for GANConfig {
    fn default() -> Self {
        Self {
            latent_dim: 100,
            generator_layers: vec![256, 512],
            discriminator_layers: vec![512, 256],
            learning_rate: 0.0002,
            n_epochs: 100,
            batch_size: 64,
            random_state: None,
        }
    }
}

impl GANConfig {
    /// Set latent dimension
    pub fn latent_dim(mut self, dim: usize) -> Self {
        self.latent_dim = dim;
        self
    }

    /// Set generator layer sizes
    pub fn generator_layers(mut self, layers: Vec<usize>) -> Self {
        self.generator_layers = layers;
        self
    }

    /// Set discriminator layer sizes
    pub fn discriminator_layers(mut self, layers: Vec<usize>) -> Self {
        self.discriminator_layers = layers;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
}

/// Simple fully connected neural network for GAN components
#[derive(Debug, Clone)]
struct SimpleNetwork {
    layers: Vec<(Array2<f64>, Array1<f64>)>, // (weights, biases)
    layer_sizes: Vec<usize>,
}

impl SimpleNetwork {
    /// Create new network
    fn new(layer_sizes: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        let mut rng = thread_rng();

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Xavier initialization
            let scale = (2.0 / (input_size + output_size) as f64).sqrt();
            let weights =
                Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-scale..scale));
            let biases = Array1::zeros(output_size);

            layers.push((weights, biases));
        }

        Self {
            layers,
            layer_sizes,
        }
    }

    /// Forward pass
    fn forward(&self, input: &Array2<f64>, use_tanh_output: bool) -> Array2<f64> {
        let mut current = input.clone();

        for (i, (weights, biases)) in self.layers.iter().enumerate() {
            current = current.dot(weights);

            // Add biases
            for mut row in current.rows_mut() {
                row += biases;
            }

            // Apply activation (ReLU for hidden layers, special for output)
            if i < self.layers.len() - 1 {
                // ReLU for hidden layers
                current.mapv_inplace(|x| x.max(0.0));
            } else if use_tanh_output {
                // Tanh for generator output
                current.mapv_inplace(|x| x.tanh());
            } else {
                // Sigmoid for discriminator output
                current.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            }
        }

        current
    }
}

/// Simple GAN implementation
pub struct SimpleGAN {
    generator: SimpleNetwork,
    discriminator: SimpleNetwork,
    config: GANConfig,
    data_dim: usize,
}

impl SimpleGAN {
    /// Create new GAN
    pub fn new(config: GANConfig, data_dim: usize, _output_size: usize) -> Self {
        // Generator: latent_dim -> hidden_layers -> data_dim
        let mut gen_layers = vec![config.latent_dim];
        gen_layers.extend_from_slice(&config.generator_layers);
        gen_layers.push(data_dim);

        // Discriminator: data_dim -> hidden_layers -> 1
        let mut disc_layers = vec![data_dim];
        disc_layers.extend_from_slice(&config.discriminator_layers);
        disc_layers.push(1);

        let generator = SimpleNetwork::new(gen_layers);
        let discriminator = SimpleNetwork::new(disc_layers);

        Self {
            generator,
            discriminator,
            config,
            data_dim,
        }
    }

    /// Generate fake samples from noise
    pub fn generate(&self, noise: &Array2<f64>) -> Array2<f64> {
        self.generator.forward(noise, true) // Use tanh output
    }

    /// Discriminate real vs fake samples
    pub fn discriminate(&self, data: &Array2<f64>) -> Array2<f64> {
        self.discriminator.forward(data, false) // Use sigmoid output
    }

    /// Generate random noise
    pub fn generate_noise(&self, batch_size: usize) -> NeuralResult<Array2<f64>> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).map_err(|_| SklearsError::InvalidParameter {
            name: "noise_distribution".to_string(),
            reason: "Failed to create normal distribution".to_string(),
        })?;

        let mut noise = Array2::zeros((batch_size, self.config.latent_dim));
        for mut row in noise.rows_mut() {
            for elem in row.iter_mut() {
                *elem = normal.sample(&mut rng);
            }
        }

        Ok(noise)
    }

    /// Train GAN on data (simplified training loop)
    pub fn fit(&mut self, data: &Array2<f64>) -> NeuralResult<GANTrainingHistory> {
        let batch_size = self.config.batch_size;
        let n_batches = data.nrows().div_ceil(batch_size);
        let mut history = GANTrainingHistory::new();

        for epoch in 0..self.config.n_epochs {
            let mut generator_losses = Vec::new();
            let mut discriminator_losses = Vec::new();

            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(data.nrows());
                let current_batch_size = end_idx - start_idx;

                // Get real data batch
                let real_data = data
                    .slice(scirs2_core::ndarray::s![start_idx..end_idx, ..])
                    .to_owned();

                // Train discriminator
                let d_loss = self.train_discriminator_step(&real_data, current_batch_size)?;
                discriminator_losses.push(d_loss);

                // Train generator
                let g_loss = self.train_generator_step(current_batch_size)?;
                generator_losses.push(g_loss);
            }

            // Record epoch metrics
            let avg_g_loss = generator_losses.iter().sum::<f64>() / generator_losses.len() as f64;
            let avg_d_loss =
                discriminator_losses.iter().sum::<f64>() / discriminator_losses.len() as f64;

            history.generator_losses.push(avg_g_loss);
            history.discriminator_losses.push(avg_d_loss);
            history.epochs_completed = epoch + 1;

            // Print progress every 10 epochs
            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: G_loss = {:.4}, D_loss = {:.4}",
                    epoch, avg_g_loss, avg_d_loss
                );
            }
        }

        Ok(history)
    }

    /// Train discriminator for one step (simplified)
    fn train_discriminator_step(
        &mut self,
        real_data: &Array2<f64>,
        batch_size: usize,
    ) -> NeuralResult<f64> {
        // Generate fake data
        let noise = self.generate_noise(batch_size)?;
        let fake_data = self.generate(&noise);

        // Get discriminator predictions
        let real_pred = self.discriminate(real_data);
        let fake_pred = self.discriminate(&fake_data);

        // Compute discriminator loss (binary cross entropy)
        let real_loss = real_pred.iter().map(|&x| -(x.max(1e-7).ln())).sum::<f64>();
        let fake_loss = fake_pred
            .iter()
            .map(|&x| -((1.0 - x).max(1e-7).ln()))
            .sum::<f64>();

        let total_loss = (real_loss + fake_loss) / (batch_size * 2) as f64;

        // Note: Actual parameter updates would require backpropagation
        // For now, just return the loss for monitoring

        Ok(total_loss)
    }

    /// Train generator for one step (simplified)
    fn train_generator_step(&mut self, batch_size: usize) -> NeuralResult<f64> {
        // Generate fake data
        let noise = self.generate_noise(batch_size)?;
        let fake_data = self.generate(&noise);

        // Get discriminator prediction on fake data
        let fake_pred = self.discriminate(&fake_data);

        // Generator wants discriminator to classify fake as real
        let generator_loss =
            fake_pred.iter().map(|&x| -(x.max(1e-7).ln())).sum::<f64>() / batch_size as f64;

        // Note: Actual parameter updates would require backpropagation
        Ok(generator_loss)
    }

    /// Generate samples from trained GAN
    pub fn generate_samples(&self, n_samples: usize) -> NeuralResult<Array2<f64>> {
        let noise = self.generate_noise(n_samples)?;
        Ok(self.generate(&noise))
    }
}

/// Training history for GAN
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GANTrainingHistory {
    pub generator_losses: Vec<f64>,
    pub discriminator_losses: Vec<f64>,
    pub epochs_completed: usize,
}

impl GANTrainingHistory {
    pub fn new() -> Self {
        Self {
            generator_losses: Vec::new(),
            discriminator_losses: Vec::new(),
            epochs_completed: 0,
        }
    }

    /// Get the final generator loss
    pub fn final_generator_loss(&self) -> Option<f64> {
        self.generator_losses.last().copied()
    }

    /// Get the final discriminator loss
    pub fn final_discriminator_loss(&self) -> Option<f64> {
        self.discriminator_losses.last().copied()
    }

    /// Check if training converged (losses stabilized)
    pub fn has_converged(&self, window_size: usize, threshold: f64) -> bool {
        if self.generator_losses.len() < window_size * 2 {
            return false;
        }

        let recent_losses = &self.generator_losses[self.generator_losses.len() - window_size..];
        let variance = Self::variance(recent_losses);
        variance < threshold
    }

    fn variance(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
    }
}

impl Default for GANTrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for GAN operations
pub mod utils {
    use super::*;

    /// Evaluate GAN quality using a simple diversity metric
    pub fn evaluate_diversity(generated_samples: &Array2<f64>) -> f64 {
        let n_samples = generated_samples.nrows();
        if n_samples < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        // Compute pairwise distances (simplified diversity measure)
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let row_i = generated_samples.row(i);
                let row_j = generated_samples.row(j);
                let distance = row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                total_distance += distance;
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }

    /// Create a simple GAN for MNIST-like data
    pub fn mnist_gan_config() -> GANConfig {
        GANConfig::default()
            .latent_dim(100)
            .generator_layers(vec![256, 512])
            .discriminator_layers(vec![512, 256])
            .learning_rate(0.0002)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gan_config() {
        let config = GANConfig::default().latent_dim(128).learning_rate(0.001);

        assert_eq!(config.latent_dim, 128);
        assert_abs_diff_eq!(config.learning_rate, 0.001, epsilon = 1e-10);
    }

    #[test]
    fn test_simple_network_creation() {
        let network = SimpleNetwork::new(vec![100, 256, 1]);
        assert_eq!(network.layer_sizes, vec![100, 256, 1]);
        assert_eq!(network.layers.len(), 2);
    }

    #[test]
    fn test_simple_network_forward() {
        let network = SimpleNetwork::new(vec![2, 3, 1]);
        let input = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let output = network.forward(&input, false);
        assert_eq!(output.shape(), &[1, 1]);
    }

    #[test]
    fn test_gan_creation() {
        let config = GANConfig::default();
        let gan = SimpleGAN::new(config, 784, 28);
        assert_eq!(gan.data_dim, 784);
    }

    #[test]
    fn test_noise_generation() {
        let config = GANConfig::default().latent_dim(100);
        let gan = SimpleGAN::new(config, 784, 28);
        let noise = gan.generate_noise(32).unwrap();

        assert_eq!(noise.shape(), &[32, 100]);
    }

    #[test]
    fn test_generation() {
        let config = GANConfig::default();
        let gan = SimpleGAN::new(config, 10, 5);
        let noise = gan.generate_noise(5).unwrap();
        let generated = gan.generate(&noise);

        assert_eq!(generated.shape(), &[5, 10]);
        // Check that tanh output is in [-1, 1]
        for &val in generated.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_discrimination() {
        let config = GANConfig::default();
        let gan = SimpleGAN::new(config, 10, 5);
        let data = Array2::ones((5, 10));
        let predictions = gan.discriminate(&data);

        assert_eq!(predictions.shape(), &[5, 1]);
        // Check that sigmoid output is in [0, 1]
        for &val in predictions.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_training_history() {
        let mut history = GANTrainingHistory::new();
        history.generator_losses.push(1.0);
        history.discriminator_losses.push(0.8);
        history.epochs_completed = 1;

        assert_eq!(history.final_generator_loss(), Some(1.0));
        assert_eq!(history.final_discriminator_loss(), Some(0.8));
    }

    #[test]
    fn test_utility_functions() {
        let samples = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let diversity = utils::evaluate_diversity(&samples);
        assert!(diversity > 0.0);

        let config = utils::mnist_gan_config();
        assert_eq!(config.latent_dim, 100);
    }
}
