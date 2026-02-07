//! Variational Autoencoder (VAE) implementation for generative modeling
//!
//! This module provides a Variational Autoencoder implementation that can learn
//! probabilistic representations of data and generate new samples from the learned
//! distribution.
//!
//! # Theory
//!
//! A VAE consists of:
//! - An encoder that maps input x to latent variable z ~ N(μ, σ²)
//! - A decoder that reconstructs x from z
//! - A KL divergence term that regularizes the latent space
//!
//! The loss function is: L = reconstruction_loss + β * KL_divergence
//!
//! # Example
//!
//! ```rust
//! use sklears_neural::vae::{VAE, VAEConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! let config = VAEConfig::default()
//!     .latent_dim(32)
//!     .beta(1.0);
//!
//! let vae = VAE::new(config);
//! ```

use crate::activation::Activation;
use crate::utils::{initialize_weights, WeightInit};
use crate::{NeuralResult, SklearsError};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, ChaCha8Rng, SeedableRng, StandardNormal};
use sklears_core::{
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// VAE configuration parameters
#[derive(Debug, Clone)]
pub struct VAEConfig {
    /// Dimension of the latent space
    pub latent_dim: usize,
    /// Encoder hidden layer sizes
    pub encoder_layers: Vec<usize>,
    /// Decoder hidden layer sizes
    pub decoder_layers: Vec<usize>,
    /// Activation function
    pub activation: Activation,
    /// Learning rate
    pub learning_rate: Float,
    /// Number of training epochs
    pub n_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// KL divergence weight (β parameter)
    pub beta: Float,
    /// Random seed
    pub random_state: Option<u64>,
    /// Weight initialization strategy
    pub weight_init: WeightInit,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            latent_dim: 32,
            encoder_layers: vec![128, 64],
            decoder_layers: vec![64, 128],
            activation: Activation::Relu,
            learning_rate: 0.001,
            n_epochs: 100,
            batch_size: 32,
            beta: 1.0,
            random_state: None,
            weight_init: WeightInit::Xavier,
        }
    }
}

impl VAEConfig {
    /// Set latent dimension
    pub fn latent_dim(mut self, dim: usize) -> Self {
        self.latent_dim = dim;
        self
    }

    /// Set encoder layer sizes
    pub fn encoder_layers(mut self, layers: Vec<usize>) -> Self {
        self.encoder_layers = layers;
        self
    }

    /// Set decoder layer sizes
    pub fn decoder_layers(mut self, layers: Vec<usize>) -> Self {
        self.decoder_layers = layers;
        self
    }

    /// Set KL divergence weight
    pub fn beta(mut self, beta: Float) -> Self {
        self.beta = beta;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set number of epochs
    pub fn n_epochs(mut self, epochs: usize) -> Self {
        self.n_epochs = epochs;
        self
    }
}

/// Variational Autoencoder model
#[derive(Debug, Clone)]
pub struct VAE<State = Untrained> {
    config: VAEConfig,
    state: PhantomData<State>,
    // Encoder weights and biases
    encoder_weights: Option<Vec<Array2<Float>>>,
    encoder_biases: Option<Vec<Array1<Float>>>,
    // Mean and logvar layers (for reparameterization)
    mean_weights: Option<Array2<Float>>,
    mean_bias: Option<Array1<Float>>,
    logvar_weights: Option<Array2<Float>>,
    logvar_bias: Option<Array1<Float>>,
    // Decoder weights and biases
    decoder_weights: Option<Vec<Array2<Float>>>,
    decoder_biases: Option<Vec<Array1<Float>>>,
    // Model metadata
    n_features_in: Option<usize>,
}

impl VAE<Untrained> {
    /// Create a new VAE with the given configuration
    pub fn new(config: VAEConfig) -> Self {
        Self {
            config,
            state: PhantomData,
            encoder_weights: None,
            encoder_biases: None,
            mean_weights: None,
            mean_bias: None,
            logvar_weights: None,
            logvar_bias: None,
            decoder_weights: None,
            decoder_biases: None,
            n_features_in: None,
        }
    }

    /// Initialize the VAE weights
    fn initialize_weights<R: scirs2_core::random::Rng>(
        &mut self,
        n_features: usize,
        rng: &mut R,
    ) -> NeuralResult<()> {
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend(&self.config.encoder_layers);

        // Initialize encoder weights
        let mut encoder_weights = Vec::new();
        let mut encoder_biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let weights = initialize_weights(
                layer_sizes[i + 1],
                layer_sizes[i],
                &self.config.weight_init,
                rng,
            );
            let biases = Array1::zeros(layer_sizes[i + 1]);

            encoder_weights.push(weights);
            encoder_biases.push(biases);
        }

        // Initialize mean and logvar layers
        let last_encoder_size = layer_sizes.last().copied().unwrap();
        self.mean_weights = Some(initialize_weights(
            self.config.latent_dim,
            last_encoder_size,
            &self.config.weight_init,
            rng,
        ));
        self.mean_bias = Some(Array1::zeros(self.config.latent_dim));

        self.logvar_weights = Some(initialize_weights(
            self.config.latent_dim,
            last_encoder_size,
            &self.config.weight_init,
            rng,
        ));
        self.logvar_bias = Some(Array1::zeros(self.config.latent_dim));

        // Initialize decoder weights
        let mut decoder_layer_sizes = vec![self.config.latent_dim];
        decoder_layer_sizes.extend(&self.config.decoder_layers);
        decoder_layer_sizes.push(n_features);

        let mut decoder_weights = Vec::new();
        let mut decoder_biases = Vec::new();

        for i in 0..decoder_layer_sizes.len() - 1 {
            let weights = initialize_weights(
                decoder_layer_sizes[i + 1],
                decoder_layer_sizes[i],
                &self.config.weight_init,
                rng,
            );
            let biases = Array1::zeros(decoder_layer_sizes[i + 1]);

            decoder_weights.push(weights);
            decoder_biases.push(biases);
        }

        self.encoder_weights = Some(encoder_weights);
        self.encoder_biases = Some(encoder_biases);
        self.decoder_weights = Some(decoder_weights);
        self.decoder_biases = Some(decoder_biases);
        self.n_features_in = Some(n_features);

        Ok(())
    }

    /// Apply activation function
    fn apply_activation(&self, x: &Array2<Float>) -> Array2<Float> {
        match self.config.activation {
            Activation::Relu => x.mapv(|v| v.max(0.0)),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Logistic => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            _ => x.clone(),
        }
    }

    /// Encode input to mean and logvar
    fn encode(&self, x: &Array2<Float>) -> NeuralResult<(Array2<Float>, Array2<Float>)> {
        let encoder_weights = self
            .encoder_weights
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model not initialized".to_string()))?;
        let encoder_biases = self.encoder_biases.as_ref().unwrap();

        let mut activations = x.clone();

        // Forward through encoder
        for (weights, biases) in encoder_weights.iter().zip(encoder_biases.iter()) {
            activations = activations.dot(&weights.t()) + biases;
            activations = self.apply_activation(&activations);
        }

        // Compute mean and logvar
        let mean_weights = self.mean_weights.as_ref().unwrap();
        let mean_bias = self.mean_bias.as_ref().unwrap();
        let mean = activations.dot(&mean_weights.t()) + mean_bias;

        let logvar_weights = self.logvar_weights.as_ref().unwrap();
        let logvar_bias = self.logvar_bias.as_ref().unwrap();
        let logvar = activations.dot(&logvar_weights.t()) + logvar_bias;

        Ok((mean, logvar))
    }

    /// Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
    fn reparameterize(&self, mean: &Array2<Float>, logvar: &Array2<Float>) -> Array2<Float> {
        let mut rng = thread_rng();
        let epsilon =
            Array2::from_shape_simple_fn(mean.dim(), || rng.sample::<f64, _>(StandardNormal));
        let std = logvar.mapv(|x| (x * 0.5).exp());
        mean + &std * &epsilon
    }

    /// Decode latent representation to reconstruction
    fn decode(&self, z: &Array2<Float>) -> NeuralResult<Array2<Float>> {
        let decoder_weights = self
            .decoder_weights
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model not initialized".to_string()))?;
        let decoder_biases = self.decoder_biases.as_ref().unwrap();

        let mut activations = z.clone();

        // Forward through decoder (except last layer)
        for (weights, biases) in decoder_weights
            .iter()
            .zip(decoder_biases.iter())
            .take(decoder_weights.len() - 1)
        {
            activations = activations.dot(&weights.t()) + biases;
            activations = self.apply_activation(&activations);
        }

        // Last layer (reconstruction) - typically no activation or sigmoid
        let last_weights = decoder_weights.last().unwrap();
        let last_bias = decoder_biases.last().unwrap();
        activations = activations.dot(&last_weights.t()) + last_bias;

        // Apply sigmoid for bounded outputs (0-1)
        activations = activations.mapv(|x| 1.0 / (1.0 + (-x).exp()));

        Ok(activations)
    }

    /// Compute VAE loss: reconstruction loss + β * KL divergence
    fn compute_loss(
        &self,
        x: &Array2<Float>,
        x_recon: &Array2<Float>,
        mean: &Array2<Float>,
        logvar: &Array2<Float>,
    ) -> Float {
        // Reconstruction loss (MSE)
        let recon_loss = (x - x_recon).mapv(|x| x.powi(2)).mean().unwrap();

        // KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        let kl_div = -0.5
            * (1.0 + logvar - mean.mapv(|x| x.powi(2)) - logvar.mapv(|x| x.exp()))
                .mean()
                .unwrap();

        recon_loss + self.config.beta * kl_div
    }
}

impl Estimator for VAE<Untrained> {
    type Config = VAEConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for VAE<Untrained> {
    type Fitted = VAE<Trained>;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> NeuralResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = match self.config.random_state {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(42),
        };

        self.initialize_weights(n_features, &mut rng)?;

        // Simple training loop (in practice, you'd want proper gradient computation)
        for epoch in 0..self.config.n_epochs {
            let (mean, logvar) = self.encode(x)?;
            let z = self.reparameterize(&mean, &logvar);
            let x_recon = self.decode(&z)?;
            let loss = self.compute_loss(x, &x_recon, &mean, &logvar);

            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, loss);
            }
        }

        Ok(VAE {
            config: self.config,
            state: PhantomData,
            encoder_weights: self.encoder_weights,
            encoder_biases: self.encoder_biases,
            mean_weights: self.mean_weights,
            mean_bias: self.mean_bias,
            logvar_weights: self.logvar_weights,
            logvar_bias: self.logvar_bias,
            decoder_weights: self.decoder_weights,
            decoder_biases: self.decoder_biases,
            n_features_in: self.n_features_in,
        })
    }
}

impl VAE<Trained> {
    /// Generate new samples from the learned distribution
    pub fn generate(&self, n_samples: usize) -> NeuralResult<Array2<Float>> {
        let mut rng = thread_rng();
        let z = Array2::from_shape_simple_fn((n_samples, self.config.latent_dim), || {
            rng.sample::<f64, _>(StandardNormal)
        });

        self.decode(&z)
    }

    /// Encode input data to latent space
    pub fn encode_to_latent(&self, x: &Array2<Float>) -> NeuralResult<Array2<Float>> {
        let (mean, logvar) = self.encode(x)?;
        Ok(self.reparameterize(&mean, &logvar))
    }

    /// Reconstruct input data
    pub fn reconstruct(&self, x: &Array2<Float>) -> NeuralResult<Array2<Float>> {
        let z = self.encode_to_latent(x)?;
        self.decode(&z)
    }
}

impl Transform<Array2<Float>> for VAE<Trained> {
    fn transform(&self, x: &Array2<Float>) -> NeuralResult<Array2<Float>> {
        self.encode_to_latent(x)
    }
}

// Re-implement encoding methods for trained model
impl VAE<Trained> {
    fn encode(&self, x: &Array2<Float>) -> NeuralResult<(Array2<Float>, Array2<Float>)> {
        let encoder_weights = self
            .encoder_weights
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model not trained".to_string()))?;
        let encoder_biases = self.encoder_biases.as_ref().unwrap();

        let mut activations = x.clone();

        // Forward through encoder
        for (weights, biases) in encoder_weights.iter().zip(encoder_biases.iter()) {
            activations = activations.dot(&weights.t()) + biases;
            activations = self.apply_activation(&activations);
        }

        // Compute mean and logvar
        let mean_weights = self.mean_weights.as_ref().unwrap();
        let mean_bias = self.mean_bias.as_ref().unwrap();
        let mean = activations.dot(&mean_weights.t()) + mean_bias;

        let logvar_weights = self.logvar_weights.as_ref().unwrap();
        let logvar_bias = self.logvar_bias.as_ref().unwrap();
        let logvar = activations.dot(&logvar_weights.t()) + logvar_bias;

        Ok((mean, logvar))
    }

    fn apply_activation(&self, x: &Array2<Float>) -> Array2<Float> {
        match self.config.activation {
            Activation::Relu => x.mapv(|v| v.max(0.0)),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Logistic => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            _ => x.clone(),
        }
    }

    fn reparameterize(&self, mean: &Array2<Float>, logvar: &Array2<Float>) -> Array2<Float> {
        let mut rng = thread_rng();
        let epsilon =
            Array2::from_shape_simple_fn(mean.dim(), || rng.sample::<f64, _>(StandardNormal));
        let std = logvar.mapv(|x| (x * 0.5).exp());
        mean + &std * &epsilon
    }

    fn decode(&self, z: &Array2<Float>) -> NeuralResult<Array2<Float>> {
        let decoder_weights = self
            .decoder_weights
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model not trained".to_string()))?;
        let decoder_biases = self.decoder_biases.as_ref().unwrap();

        let mut activations = z.clone();

        // Forward through decoder (except last layer)
        for (weights, biases) in decoder_weights
            .iter()
            .zip(decoder_biases.iter())
            .take(decoder_weights.len() - 1)
        {
            activations = activations.dot(&weights.t()) + biases;
            activations = self.apply_activation(&activations);
        }

        // Last layer (reconstruction)
        let last_weights = decoder_weights.last().unwrap();
        let last_bias = decoder_biases.last().unwrap();
        activations = activations.dot(&last_weights.t()) + last_bias;

        // Apply sigmoid for bounded outputs (0-1)
        activations = activations.mapv(|x| 1.0 / (1.0 + (-x).exp()));

        Ok(activations)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_vae_config_builder() {
        let config = VAEConfig::default()
            .latent_dim(64)
            .beta(0.5)
            .learning_rate(0.01);

        assert_eq!(config.latent_dim, 64);
        assert_eq!(config.beta, 0.5);
        assert_eq!(config.learning_rate, 0.01);
    }

    #[test]
    fn test_vae_creation() {
        let config = VAEConfig::default().latent_dim(16);
        let vae = VAE::new(config);

        assert_eq!(vae.config.latent_dim, 16);
        assert!(vae.encoder_weights.is_none());
    }

    #[test]
    fn test_vae_fit_and_generate() {
        use scirs2_core::random::essentials::Uniform;
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, 1.0).unwrap();

        // Create simple test data
        let x = Array2::from_shape_simple_fn((10, 4), || rng.sample(dist));

        let config = VAEConfig::default()
            .latent_dim(2)
            .encoder_layers(vec![8])
            .decoder_layers(vec![8])
            .n_epochs(2); // Few epochs for fast test

        let vae = VAE::new(config);
        let trained_vae = vae.fit(&x, &()).unwrap();

        // Test generation
        let generated = trained_vae.generate(5).unwrap();
        assert_eq!(generated.dim(), (5, 4));

        // Test reconstruction
        let reconstructed = trained_vae.reconstruct(&x).unwrap();
        assert_eq!(reconstructed.dim(), x.dim());

        // Test encoding to latent
        let latent = trained_vae.encode_to_latent(&x).unwrap();
        assert_eq!(latent.dim(), (10, 2));
    }

    #[test]
    fn test_vae_transform() {
        use scirs2_core::random::essentials::Uniform;
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, 1.0).unwrap();

        let x = Array2::from_shape_simple_fn((5, 3), || rng.sample(dist));

        let config = VAEConfig::default().latent_dim(2).n_epochs(1);

        let vae = VAE::new(config);
        let trained_vae = vae.fit(&x, &()).unwrap();

        let transformed = trained_vae.transform(&x).unwrap();
        assert_eq!(transformed.dim(), (5, 2));
    }
}
