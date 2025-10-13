//! Variational Autoencoder for Cross-Modal Learning
//!
//! This module provides VAE-based cross-decomposition methods for learning
//! shared representations between different data modalities.
//!
//! ## Applications
//! - Cross-modal learning between text and images
//! - Multi-view representation learning
//! - Shared latent space discovery
//! - Cross-modal retrieval and generation

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Distribution;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, Random};
use sklears_core::error::SklearsError;
use sklears_core::types::Float;
use std::collections::HashMap;

/// Configuration for Variational Autoencoder
#[derive(Debug, Clone)]
pub struct VAEConfig {
    /// Latent space dimensions
    pub latent_dim: usize,
    /// Hidden layer dimensions for encoder
    pub encoder_hidden_dims: Vec<usize>,
    /// Hidden layer dimensions for decoder
    pub decoder_hidden_dims: Vec<usize>,
    /// Learning rate for optimization
    pub learning_rate: Float,
    /// Beta parameter for KL divergence weighting
    pub beta: Float,
    /// Maximum training epochs
    pub max_epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Use batch normalization
    pub use_batch_norm: bool,
    /// Dropout probability
    pub dropout_prob: Float,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            latent_dim: 10,
            encoder_hidden_dims: vec![128, 64],
            decoder_hidden_dims: vec![64, 128],
            learning_rate: 1e-3,
            beta: 1.0,
            max_epochs: 100,
            batch_size: 32,
            patience: 10,
            tolerance: 1e-6,
            use_batch_norm: true,
            dropout_prob: 0.2,
        }
    }
}

/// Activation functions for neural networks
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    /// ReLU
    ReLU,
    /// Sigmoid
    Sigmoid,
    /// Tanh
    Tanh,
    /// LeakyReLU
    LeakyReLU(Float),
    /// Swish
    Swish,
    /// GELU
    GELU,
}

impl ActivationFunction {
    /// Apply activation function
    pub fn apply(&self, x: &Array2<Float>) -> Array2<Float> {
        match self {
            Self::ReLU => x.mapv(|v| v.max(0.0)),
            Self::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Self::Tanh => x.mapv(|v| v.tanh()),
            Self::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { v } else { *alpha * v }),
            Self::Swish => x.mapv(|v| v / (1.0 + (-v).exp())),
            Self::GELU => x.mapv(|v| 0.5 * v * (1.0 + (v * 0.7978845608).tanh())),
        }
    }

    /// Compute derivative of activation function
    pub fn derivative(&self, x: &Array2<Float>) -> Array2<Float> {
        match self {
            Self::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Self::Sigmoid => {
                let sigmoid = self.apply(x);
                &sigmoid * &sigmoid.mapv(|v| 1.0 - v)
            }
            Self::Tanh => {
                let tanh = self.apply(x);
                tanh.mapv(|v| 1.0 - v * v)
            }
            Self::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { 1.0 } else { *alpha }),
            Self::Swish => {
                let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                let swish = self.apply(x);
                &sigmoid * &(swish.mapv(|s| 1.0) + &(x * &sigmoid.mapv(|s| 1.0 - s)))
            }
            Self::GELU => {
                // Approximate GELU derivative
                x.mapv(|v| {
                    let phi = 0.5 * (1.0 + (v * 0.7978845608).tanh());
                    phi + v * 0.5 * (1.0 - (v * 0.7978845608).tanh().powi(2)) * 0.7978845608
                })
            }
        }
    }
}

/// Neural network layer
#[derive(Debug, Clone)]
pub struct Layer {
    /// Weight matrix
    pub weights: Array2<Float>,
    /// Bias vector
    pub bias: Array1<Float>,
    /// Activation function
    pub activation: ActivationFunction,
    /// Use batch normalization
    pub use_batch_norm: bool,
    /// Batch normalization parameters
    pub batch_norm_gamma: Option<Array1<Float>>,
    pub batch_norm_beta: Option<Array1<Float>>,
}

impl Layer {
    /// Create new layer
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        activation: ActivationFunction,
        use_batch_norm: bool,
    ) -> Self {
        // Xavier initialization
        let scale = (2.0 / input_dim as Float).sqrt();
        let mut rng = thread_rng();
        let weights = Array2::<Float>::from_shape_fn((input_dim, output_dim), |_| {
            (rng.gen::<Float>() - 0.5) * 2.0 * scale
        });
        let bias = Array1::<Float>::zeros(output_dim);

        let (batch_norm_gamma, batch_norm_beta) = if use_batch_norm {
            (
                Some(Array1::<Float>::ones(output_dim)),
                Some(Array1::<Float>::zeros(output_dim)),
            )
        } else {
            (None, None)
        };

        Self {
            weights,
            bias,
            activation,
            use_batch_norm,
            batch_norm_gamma,
            batch_norm_beta,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array2<Float>) -> Array2<Float> {
        // Linear transformation
        let linear_output = input.dot(&self.weights) + &self.bias.view().insert_axis(Axis(0));

        // Batch normalization if enabled
        let normalized = if self.use_batch_norm {
            self.batch_normalize(&linear_output)
        } else {
            linear_output
        };

        // Apply activation
        self.activation.apply(&normalized)
    }

    /// Apply batch normalization
    fn batch_normalize(&self, input: &Array2<Float>) -> Array2<Float> {
        if let (Some(gamma), Some(beta)) = (&self.batch_norm_gamma, &self.batch_norm_beta) {
            let mean = input.mean_axis(Axis(0)).unwrap();
            let var = input.var_axis(Axis(0), 0.0);
            let eps = 1e-8;

            let normalized = (input - &mean.view().insert_axis(Axis(0)))
                / &(var + eps).mapv(|v| v.sqrt()).view().insert_axis(Axis(0));

            &normalized * &gamma.view().insert_axis(Axis(0)) + &beta.view().insert_axis(Axis(0))
        } else {
            input.clone()
        }
    }
}

/// Encoder network
#[derive(Debug, Clone)]
pub struct Encoder {
    /// Hidden layers
    layers: Vec<Layer>,
    /// Mean projection layer
    mean_layer: Layer,
    /// Log variance projection layer
    logvar_layer: Layer,
}

impl Encoder {
    /// Create new encoder
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        latent_dim: usize,
        use_batch_norm: bool,
    ) -> Self {
        let mut layers = Vec::new();
        let mut current_dim = input_dim;

        // Hidden layers
        for &hidden_dim in hidden_dims {
            layers.push(Layer::new(
                current_dim,
                hidden_dim,
                ActivationFunction::ReLU,
                use_batch_norm,
            ));
            current_dim = hidden_dim;
        }

        // Mean and log variance layers (no activation for final projections)
        let mean_layer = Layer::new(current_dim, latent_dim, ActivationFunction::ReLU, false);
        let logvar_layer = Layer::new(current_dim, latent_dim, ActivationFunction::ReLU, false);

        Self {
            layers,
            mean_layer,
            logvar_layer,
        }
    }

    /// Encode input to latent distribution parameters
    pub fn encode(&self, input: &Array2<Float>) -> (Array2<Float>, Array2<Float>) {
        let mut hidden = input.clone();

        // Forward through hidden layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden);
        }

        // Get mean and log variance
        let mean = self.mean_layer.forward(&hidden);
        let logvar = self.logvar_layer.forward(&hidden);

        (mean, logvar)
    }
}

/// Decoder network
#[derive(Debug, Clone)]
pub struct Decoder {
    /// Layers
    layers: Vec<Layer>,
}

impl Decoder {
    /// Create new decoder
    pub fn new(
        latent_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        use_batch_norm: bool,
    ) -> Self {
        let mut layers = Vec::new();
        let mut current_dim = latent_dim;

        // Hidden layers
        for &hidden_dim in hidden_dims {
            layers.push(Layer::new(
                current_dim,
                hidden_dim,
                ActivationFunction::ReLU,
                use_batch_norm,
            ));
            current_dim = hidden_dim;
        }

        // Output layer (usually sigmoid for bounded outputs)
        layers.push(Layer::new(
            current_dim,
            output_dim,
            ActivationFunction::Sigmoid,
            false,
        ));

        Self { layers }
    }

    /// Decode latent vector to output
    pub fn decode(&self, latent: &Array2<Float>) -> Array2<Float> {
        let mut output = latent.clone();

        for layer in &self.layers {
            output = layer.forward(&output);
        }

        output
    }
}

/// Variational Autoencoder for cross-modal learning
#[derive(Debug, Clone)]
pub struct CrossModalVAE {
    /// Encoder for modality X
    encoder_x: Encoder,
    /// Encoder for modality Y
    encoder_y: Encoder,
    /// Shared decoder for both modalities
    decoder_x: Decoder,
    /// Decoder for modality Y
    decoder_y: Decoder,
    /// Configuration
    config: VAEConfig,
    /// Training history
    training_history: HashMap<String, Vec<Float>>,
}

impl CrossModalVAE {
    /// Create new cross-modal VAE
    pub fn new(input_dim_x: usize, input_dim_y: usize, config: VAEConfig) -> Self {
        let encoder_x = Encoder::new(
            input_dim_x,
            &config.encoder_hidden_dims,
            config.latent_dim,
            config.use_batch_norm,
        );

        let encoder_y = Encoder::new(
            input_dim_y,
            &config.encoder_hidden_dims,
            config.latent_dim,
            config.use_batch_norm,
        );

        let decoder_x = Decoder::new(
            config.latent_dim,
            &config.decoder_hidden_dims,
            input_dim_x,
            config.use_batch_norm,
        );

        let decoder_y = Decoder::new(
            config.latent_dim,
            &config.decoder_hidden_dims,
            input_dim_y,
            config.use_batch_norm,
        );

        Self {
            encoder_x,
            encoder_y,
            decoder_x,
            decoder_y,
            config,
            training_history: HashMap::new(),
        }
    }

    /// Train the VAE on cross-modal data
    pub fn fit(
        &mut self,
        x_data: &Array2<Float>,
        y_data: &Array2<Float>,
    ) -> Result<VAETrainingResults, SklearsError> {
        if x_data.nrows() != y_data.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y data must have same number of samples".to_string(),
            ));
        }

        let n_samples = x_data.nrows();
        let mut losses = Vec::new();
        let mut kl_losses = Vec::new();
        let mut recon_losses = Vec::new();

        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;
            let mut epoch_kl = 0.0;
            let mut epoch_recon = 0.0;
            let mut n_batches = 0;

            // Mini-batch training
            for i in (0..n_samples).step_by(self.config.batch_size) {
                let end = (i + self.config.batch_size).min(n_samples);
                let batch_x = x_data.slice(s![i..end, ..]);
                let batch_y = y_data.slice(s![i..end, ..]);

                let batch_loss = self.train_batch(&batch_x, &batch_y)?;
                epoch_loss += batch_loss.total_loss;
                epoch_kl += batch_loss.kl_loss;
                epoch_recon += batch_loss.reconstruction_loss;
                n_batches += 1;
            }

            epoch_loss /= n_batches as Float;
            epoch_kl /= n_batches as Float;
            epoch_recon /= n_batches as Float;

            losses.push(epoch_loss);
            kl_losses.push(epoch_kl);
            recon_losses.push(epoch_recon);

            // Early stopping check
            if epoch > self.config.patience {
                let recent_improvement = losses[epoch - self.config.patience] - epoch_loss;
                if recent_improvement < self.config.tolerance {
                    break;
                }
            }
        }

        self.training_history
            .insert("total_loss".to_string(), losses.clone());
        self.training_history
            .insert("kl_loss".to_string(), kl_losses.clone());
        self.training_history
            .insert("reconstruction_loss".to_string(), recon_losses.clone());

        Ok(VAETrainingResults {
            final_loss: *losses.last().unwrap_or(&Float::INFINITY),
            n_epochs: losses.len(),
            training_losses: losses.clone(),
            kl_losses,
            reconstruction_losses: recon_losses,
            converged: losses.len() < self.config.max_epochs,
        })
    }

    /// Train on a single batch
    fn train_batch(
        &mut self,
        batch_x: &ArrayView2<Float>,
        batch_y: &ArrayView2<Float>,
    ) -> Result<BatchLoss, SklearsError> {
        // Encode both modalities
        let (mean_x, logvar_x) = self.encoder_x.encode(&batch_x.to_owned());
        let (mean_y, logvar_y) = self.encoder_y.encode(&batch_y.to_owned());

        // Sample from latent distributions
        let z_x = self.reparameterize(&mean_x, &logvar_x);
        let z_y = self.reparameterize(&mean_y, &logvar_y);

        // Cross-modal reconstruction
        let recon_x_from_x = self.decoder_x.decode(&z_x);
        let recon_y_from_y = self.decoder_y.decode(&z_y);
        let recon_x_from_y = self.decoder_x.decode(&z_y);
        let recon_y_from_x = self.decoder_y.decode(&z_x);

        // Compute losses
        let recon_loss_x = self.reconstruction_loss(&batch_x.to_owned(), &recon_x_from_x);
        let recon_loss_y = self.reconstruction_loss(&batch_y.to_owned(), &recon_y_from_y);
        let cross_recon_loss_x = self.reconstruction_loss(&batch_x.to_owned(), &recon_x_from_y);
        let cross_recon_loss_y = self.reconstruction_loss(&batch_y.to_owned(), &recon_y_from_x);

        let kl_loss_x = self.kl_divergence(&mean_x, &logvar_x);
        let kl_loss_y = self.kl_divergence(&mean_y, &logvar_y);

        let reconstruction_loss =
            recon_loss_x + recon_loss_y + cross_recon_loss_x + cross_recon_loss_y;
        let kl_loss = kl_loss_x + kl_loss_y;
        let total_loss = reconstruction_loss + self.config.beta * kl_loss;

        // TODO: Implement actual gradient descent updates
        // This would require automatic differentiation

        Ok(BatchLoss {
            total_loss,
            reconstruction_loss,
            kl_loss,
        })
    }

    /// Reparameterization trick for sampling
    fn reparameterize(&self, mean: &Array2<Float>, logvar: &Array2<Float>) -> Array2<Float> {
        let std = logvar.mapv(|x| (0.5 * x).exp());
        let mut rng = thread_rng();
        let epsilon = Array2::<Float>::from_shape_fn(mean.dim(), |_| {
            use scirs2_core::random::{Distribution, RandNormal as Normal};
            Normal::new(0.0, 1.0).unwrap().sample(&mut rng)
        });
        mean + &std * &epsilon
    }

    /// Compute reconstruction loss (MSE)
    fn reconstruction_loss(&self, target: &Array2<Float>, prediction: &Array2<Float>) -> Float {
        let diff = target - prediction;
        diff.mapv(|x| x * x).sum() / (target.len() as Float)
    }

    /// Compute KL divergence loss
    fn kl_divergence(&self, mean: &Array2<Float>, logvar: &Array2<Float>) -> Float {
        // KL divergence between N(mean, var) and N(0, 1)
        let kl = -0.5 * (1.0 + logvar - &mean.mapv(|x| x * x) - &logvar.mapv(|x| x.exp()));
        kl.sum() / (mean.nrows() as Float)
    }

    /// Encode data from modality X
    pub fn encode_x(&self, x: &Array2<Float>) -> Array2<Float> {
        let (mean, logvar) = self.encoder_x.encode(x);
        self.reparameterize(&mean, &logvar)
    }

    /// Encode data from modality Y
    pub fn encode_y(&self, y: &Array2<Float>) -> Array2<Float> {
        let (mean, logvar) = self.encoder_y.encode(y);
        self.reparameterize(&mean, &logvar)
    }

    /// Decode to modality X
    pub fn decode_to_x(&self, z: &Array2<Float>) -> Array2<Float> {
        self.decoder_x.decode(z)
    }

    /// Decode to modality Y
    pub fn decode_to_y(&self, z: &Array2<Float>) -> Array2<Float> {
        self.decoder_y.decode(z)
    }

    /// Cross-modal generation: X -> Y
    pub fn cross_generate_x_to_y(&self, x: &Array2<Float>) -> Array2<Float> {
        let z = self.encode_x(x);
        self.decode_to_y(&z)
    }

    /// Cross-modal generation: Y -> X
    pub fn cross_generate_y_to_x(&self, y: &Array2<Float>) -> Array2<Float> {
        let z = self.encode_y(y);
        self.decode_to_x(&z)
    }

    /// Get shared latent representation
    pub fn get_shared_representation(&self, x: &Array2<Float>, y: &Array2<Float>) -> Array2<Float> {
        let z_x = self.encode_x(x);
        let z_y = self.encode_y(y);
        (&z_x + &z_y) / 2.0 // Average of both encodings
    }

    /// Get training history
    pub fn training_history(&self) -> &HashMap<String, Vec<Float>> {
        &self.training_history
    }
}

/// Training results
#[derive(Debug, Clone)]
pub struct VAETrainingResults {
    /// Final training loss
    pub final_loss: Float,
    /// Number of epochs trained
    pub n_epochs: usize,
    /// Training loss history
    pub training_losses: Vec<Float>,
    /// KL divergence loss history
    pub kl_losses: Vec<Float>,
    /// Reconstruction loss history
    pub reconstruction_losses: Vec<Float>,
    /// Whether training converged
    pub converged: bool,
}

/// Batch training loss components
#[derive(Debug, Clone)]
struct BatchLoss {
    total_loss: Float,
    reconstruction_loss: Float,
    kl_loss: Float,
}

/// Cross-modal similarity metrics
#[derive(Debug, Clone)]
pub struct CrossModalSimilarity {
    /// Cosine similarity between modalities
    pub cosine_similarity: Float,
    /// Canonical correlation between latent representations
    pub canonical_correlation: Float,
    /// Mutual information estimate
    pub mutual_information: Float,
}

impl CrossModalVAE {
    /// Compute cross-modal similarity metrics
    pub fn compute_cross_modal_similarity(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<CrossModalSimilarity, SklearsError> {
        let z_x = self.encode_x(x);
        let z_y = self.encode_y(y);

        let cosine_similarity = self.compute_cosine_similarity(&z_x, &z_y)?;
        let canonical_correlation = self.compute_canonical_correlation(&z_x, &z_y)?;
        let mutual_information = self.estimate_mutual_information(&z_x, &z_y)?;

        Ok(CrossModalSimilarity {
            cosine_similarity,
            canonical_correlation,
            mutual_information,
        })
    }

    /// Compute average cosine similarity between latent representations
    fn compute_cosine_similarity(
        &self,
        z_x: &Array2<Float>,
        z_y: &Array2<Float>,
    ) -> Result<Float, SklearsError> {
        let mut similarities = Vec::new();

        for i in 0..z_x.nrows() {
            let x_vec = z_x.row(i);
            let y_vec = z_y.row(i);

            let dot_product = x_vec.dot(&y_vec);
            let norm_x = x_vec.mapv(|v| v * v).sum().sqrt();
            let norm_y = y_vec.mapv(|v| v * v).sum().sqrt();

            if norm_x > 1e-10 && norm_y > 1e-10 {
                similarities.push(dot_product / (norm_x * norm_y));
            }
        }

        Ok(similarities.iter().sum::<Float>() / similarities.len() as Float)
    }

    /// Compute canonical correlation between latent representations
    fn compute_canonical_correlation(
        &self,
        z_x: &Array2<Float>,
        z_y: &Array2<Float>,
    ) -> Result<Float, SklearsError> {
        // Simplified canonical correlation computation
        // In practice, would use proper CCA algorithm
        let corr_matrix = self.compute_correlation_matrix(z_x, z_y)?;
        Ok(corr_matrix.diag().sum() / corr_matrix.nrows() as Float)
    }

    /// Estimate mutual information between latent representations
    fn estimate_mutual_information(
        &self,
        z_x: &Array2<Float>,
        z_y: &Array2<Float>,
    ) -> Result<Float, SklearsError> {
        // Simplified MI estimation using correlation
        // In practice, would use KDE or neural estimation
        let correlation = self.compute_canonical_correlation(z_x, z_y)?;
        Ok(-0.5 * (1.0 - correlation * correlation).ln())
    }

    /// Compute correlation matrix between two sets of variables
    fn compute_correlation_matrix(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<Array2<Float>, SklearsError> {
        let n_samples = x.nrows() as Float;

        // Center the data
        let x_centered = x - &x.mean_axis(Axis(0)).unwrap().view().insert_axis(Axis(0));
        let y_centered = y - &y.mean_axis(Axis(0)).unwrap().view().insert_axis(Axis(0));

        // Compute covariance
        let cov = x_centered.t().dot(&y_centered) / (n_samples - 1.0);

        // Compute standard deviations
        let std_x = x_centered
            .mapv(|v| v * v)
            .sum_axis(Axis(0))
            .mapv(|v| (v / (n_samples - 1.0)).sqrt());
        let std_y = y_centered
            .mapv(|v| v * v)
            .sum_axis(Axis(0))
            .mapv(|v| (v / (n_samples - 1.0)).sqrt());

        // Normalize to get correlation
        let mut corr = cov;
        for i in 0..corr.nrows() {
            for j in 0..corr.ncols() {
                corr[[i, j]] /= std_x[i] * std_y[j];
            }
        }

        Ok(corr)
    }
}

// Import slice macro
use scirs2_core::ndarray::s;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::essentials::Normal;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(10, 5, ActivationFunction::ReLU, true);
        assert_eq!(layer.weights.dim(), (10, 5));
        assert_eq!(layer.bias.len(), 5);
        assert!(layer.batch_norm_gamma.is_some());
        assert!(layer.batch_norm_beta.is_some());
    }

    #[test]
    fn test_activation_functions() {
        let input =
            Array2::<Float>::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, 2.0, -2.0, 0.5]).unwrap();

        let relu = ActivationFunction::ReLU;
        let relu_output = relu.apply(&input);
        assert_eq!(relu_output[[0, 0]], 0.0); // ReLU(-1) = 0
        assert_eq!(relu_output[[0, 2]], 1.0); // ReLU(1) = 1

        let sigmoid = ActivationFunction::Sigmoid;
        let sigmoid_output = sigmoid.apply(&input);
        assert!(sigmoid_output[[0, 1]] > 0.4 && sigmoid_output[[0, 1]] < 0.6); // Sigmoid(0) ≈ 0.5
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = Encoder::new(20, &[16, 8], 4, true);
        assert_eq!(encoder.layers.len(), 2);
        assert_eq!(encoder.mean_layer.weights.dim(), (8, 4));
        assert_eq!(encoder.logvar_layer.weights.dim(), (8, 4));
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = Decoder::new(4, &[8, 16], 20, true);
        assert_eq!(decoder.layers.len(), 3); // 2 hidden + 1 output
    }

    #[test]
    fn test_vae_creation() {
        let config = VAEConfig::default();
        let vae = CrossModalVAE::new(28 * 28, 100, config);

        assert_eq!(vae.config.latent_dim, 10);
    }

    #[test]
    fn test_encoding_decoding() {
        let config = VAEConfig {
            latent_dim: 5,
            max_epochs: 1, // Minimal training for test
            ..VAEConfig::default()
        };
        let vae = CrossModalVAE::new(10, 8, config);

        let x = Array2::from_shape_fn((4, 10), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((4, 8), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });

        let z_x = vae.encode_x(&x);
        let z_y = vae.encode_y(&y);

        assert_eq!(z_x.dim(), (4, 5));
        assert_eq!(z_y.dim(), (4, 5));

        let recon_x = vae.decode_to_x(&z_x);
        let recon_y = vae.decode_to_y(&z_y);

        assert_eq!(recon_x.dim(), (4, 10));
        assert_eq!(recon_y.dim(), (4, 8));
    }

    #[test]
    fn test_cross_modal_generation() {
        let config = VAEConfig {
            latent_dim: 3,
            max_epochs: 1,
            ..VAEConfig::default()
        };
        let vae = CrossModalVAE::new(6, 4, config);

        let x = Array2::from_shape_fn((3, 6), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((3, 4), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });

        let y_from_x = vae.cross_generate_x_to_y(&x);
        let x_from_y = vae.cross_generate_y_to_x(&y);

        assert_eq!(y_from_x.dim(), (3, 4));
        assert_eq!(x_from_y.dim(), (3, 6));
    }

    #[test]
    fn test_shared_representation() {
        let config = VAEConfig {
            latent_dim: 4,
            max_epochs: 1,
            ..VAEConfig::default()
        };
        let vae = CrossModalVAE::new(8, 6, config);

        let x = Array2::from_shape_fn((5, 8), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((5, 6), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });

        let shared_repr = vae.get_shared_representation(&x, &y);
        assert_eq!(shared_repr.dim(), (5, 4));
    }

    #[test]
    fn test_reparameterization() {
        let config = VAEConfig::default();
        let vae = CrossModalVAE::new(10, 10, config);

        let mean = Array2::<Float>::zeros((3, 5));
        let logvar = Array2::<Float>::zeros((3, 5)); // log(1) = 0, so std = 1

        let sample = vae.reparameterize(&mean, &logvar);
        assert_eq!(sample.dim(), (3, 5));
    }

    #[test]
    fn test_loss_computation() {
        let config = VAEConfig::default();
        let vae = CrossModalVAE::new(4, 4, config);

        let target = Array2::<Float>::ones((2, 4));
        let prediction = Array2::<Float>::ones((2, 4)) * 0.5;

        let recon_loss = vae.reconstruction_loss(&target, &prediction);
        assert!(recon_loss > 0.0);

        let mean = Array2::<Float>::ones((2, 3));
        let logvar = Array2::<Float>::zeros((2, 3));
        let kl_loss = vae.kl_divergence(&mean, &logvar);
        assert!(kl_loss > 0.0);
    }

    #[test]
    fn test_cross_modal_similarity() {
        let config = VAEConfig {
            latent_dim: 3,
            max_epochs: 1,
            ..VAEConfig::default()
        };
        let vae = CrossModalVAE::new(6, 4, config);

        let x = Array2::from_shape_fn((10, 6), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((10, 4), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });

        let similarity = vae.compute_cross_modal_similarity(&x, &y);
        assert!(similarity.is_ok());

        let sim_metrics = similarity.unwrap();
        assert!(sim_metrics.cosine_similarity >= -1.0 && sim_metrics.cosine_similarity <= 1.0);
    }

    #[test]
    fn test_batch_norm() {
        let layer = Layer::new(5, 3, ActivationFunction::ReLU, true);
        let input = Array2::from_shape_fn((4, 5), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });

        let output = layer.forward(&input);
        assert_eq!(output.dim(), (4, 3));
    }
}
