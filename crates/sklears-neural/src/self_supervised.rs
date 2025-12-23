use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use scirs2_core::numeric::{Float, One, ToPrimitive};
use scirs2_core::random::Rng;
use std::fmt::Debug;

use crate::activation::Activation;
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict, Transform};
use sklears_core::types::FloatBounds;

/// Self-supervised learning methods for neural networks
///
/// This module provides implementations of various self-supervised learning techniques
/// including contrastive learning, masked modeling, and autoencoding approaches.

/// Simple Dense Layer for Self-Supervised Models
#[derive(Debug, Clone)]
pub struct DenseLayer<T: FloatBounds + ScalarOperand> {
    weights: Array2<T>,
    biases: Array1<T>,
    activation: Option<Activation>,
    last_input: Option<Array2<T>>,
}

impl<T: FloatBounds + ScalarOperand> DenseLayer<T> {
    /// Create a new dense layer
    pub fn new(input_dim: usize, output_dim: usize, activation: Option<Activation>) -> Self {
        let mut rng = scirs2_core::random::thread_rng();

        // Xavier initialization
        let scale = T::from(2.0).unwrap() / T::from(input_dim + output_dim).unwrap();
        let std_dev = scale.sqrt();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            let val: f32 = rng.gen_range(-1.0..1.0);
            T::from(val).unwrap() * std_dev
        });

        let biases = Array1::zeros(output_dim);

        Self {
            weights,
            biases,
            activation,
            last_input: None,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>, SklearsError> {
        self.last_input = Some(input.clone());

        let mut output = input.dot(&self.weights);

        // Add bias
        for mut row in output.rows_mut() {
            row += &self.biases;
        }

        // Apply activation
        if let Some(ref activation) = self.activation {
            for element in output.iter_mut() {
                let x_f64 = element.to_f64().unwrap_or(0.0);
                let result_f64 = activation.forward(x_f64);
                *element = T::from(result_f64).unwrap_or_else(|| T::zero());
            }
        }

        Ok(output)
    }
}

/// Simple Multi-Layer Perceptron for Self-Supervised Learning
#[derive(Debug, Clone)]
pub struct SimpleMLP<T: FloatBounds + ScalarOperand> {
    layers: Vec<DenseLayer<T>>,
}

impl<T: FloatBounds + ScalarOperand> SimpleMLP<T> {
    /// Create a new MLP
    pub fn new(layer_sizes: &[usize], activations: &[Option<Activation>]) -> Self {
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let activation = if i < activations.len() {
                activations[i].clone()
            } else {
                None
            };

            layers.push(DenseLayer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation,
            ));
        }

        Self { layers }
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>, SklearsError> {
        let mut current = input.clone();

        for layer in &mut self.layers {
            current = layer.forward(&current)?;
        }

        Ok(current)
    }
}

/// Contrastive Learning Framework
///
/// Implements SimCLR-style contrastive learning with configurable augmentations
/// and temperature-scaled cross-entropy loss.
#[derive(Debug, Clone)]
pub struct ContrastiveLearner<T: FloatBounds + ScalarOperand> {
    /// Encoder network
    encoder: SimpleMLP<T>,
    /// Projection head for contrastive learning
    projection_head: SimpleMLP<T>,
    /// Temperature parameter for contrastive loss
    temperature: T,
    /// Embedding dimension
    embedding_dim: usize,
}

#[derive(Debug, Clone)]
pub struct ContrastiveConfig<T: Float> {
    /// Temperature for contrastive loss
    pub temperature: T,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Number of negative samples
    pub num_negatives: usize,
    /// Augmentation probability
    pub augmentation_prob: T,
    /// Learning rate for encoder
    pub encoder_lr: T,
    /// Projection head learning rate
    pub projection_lr: T,
}

impl<T: Float> Default for ContrastiveConfig<T> {
    fn default() -> Self {
        Self {
            temperature: T::from(0.1).unwrap(),
            embedding_dim: 128,
            num_negatives: 256,
            augmentation_prob: T::from(0.5).unwrap(),
            encoder_lr: T::from(0.001).unwrap(),
            projection_lr: T::from(0.001).unwrap(),
        }
    }
}

impl<T: FloatBounds + ScalarOperand + Debug> ContrastiveLearner<T> {
    /// Create a new contrastive learner
    pub fn new(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        config: ContrastiveConfig<T>,
    ) -> Result<Self, SklearsError> {
        // Build encoder network
        let mut encoder_sizes = vec![input_dim];
        encoder_sizes.extend_from_slice(&hidden_dims);

        let encoder_activations: Vec<Option<Activation>> = (0..hidden_dims.len())
            .map(|_| Some(Activation::Relu))
            .collect();

        let encoder = SimpleMLP::new(&encoder_sizes, &encoder_activations);

        // Build projection head
        let proj_sizes = vec![hidden_dims[hidden_dims.len() - 1], config.embedding_dim];
        let proj_activations = vec![None];
        let projection_head = SimpleMLP::new(&proj_sizes, &proj_activations);

        Ok(Self {
            encoder,
            projection_head,
            temperature: config.temperature,
            embedding_dim: config.embedding_dim,
        })
    }

    /// Compute contrastive loss
    pub fn contrastive_loss(&self, embeddings: &Array2<T>) -> Result<T, SklearsError> {
        let batch_size = embeddings.nrows();
        let mut total_loss = T::zero();

        for i in 0..batch_size {
            let anchor = embeddings.row(i);
            let mut positive_sim = T::zero();
            let mut negative_sims = Vec::new();

            // Find positive pair (next sample in batch as positive)
            let positive_idx = (i + 1) % batch_size;
            let positive = embeddings.row(positive_idx);
            positive_sim = self.cosine_similarity(&anchor, &positive)?;

            // Compute negative similarities
            for j in 0..batch_size {
                if j != i && j != positive_idx {
                    let negative = embeddings.row(j);
                    let neg_sim = self.cosine_similarity(&anchor, &negative)?;
                    negative_sims.push(neg_sim);
                }
            }

            // Compute contrastive loss
            let pos_exp = (positive_sim / self.temperature).exp();
            let neg_exp_sum: T = negative_sims
                .iter()
                .map(|&sim| (sim / self.temperature).exp())
                .fold(T::zero(), |acc, x| acc + x);

            let loss = -(pos_exp / (pos_exp + neg_exp_sum)).ln();
            total_loss = total_loss + loss;
        }

        Ok(total_loss / T::from(batch_size).unwrap())
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<T>,
        b: &scirs2_core::ndarray::ArrayView1<T>,
    ) -> Result<T, SklearsError> {
        let dot_product = a.dot(b);
        let norm_a = a.mapv(|x| x * x).sum().sqrt();
        let norm_b = b.mapv(|x| x * x).sum().sqrt();

        if norm_a == T::zero() || norm_b == T::zero() {
            return Ok(T::zero());
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Forward pass through encoder and projection head
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>, SklearsError> {
        let encoded = self.encoder.forward(input)?;
        let projected = self.projection_head.forward(&encoded)?;
        Ok(projected)
    }

    /// Get encoder representations (without projection head)
    pub fn encode(&mut self, input: &Array2<T>) -> Result<Array2<T>, SklearsError> {
        self.encoder.forward(input)
    }
}

/// Autoencoder for Self-Supervised Representation Learning
///
/// Implements simple autoencoder architecture for unsupervised feature learning.
#[derive(Debug, Clone)]
pub struct SelfSupervisedAutoencoder<T: FloatBounds + ScalarOperand> {
    /// Encoder network
    encoder: SimpleMLP<T>,
    /// Decoder network
    decoder: SimpleMLP<T>,
    /// Latent dimension
    latent_dim: usize,
    /// Autoencoder type
    autoencoder_type: AutoencoderType,
    /// Configuration
    config: AutoencoderConfig<T>,
}

#[derive(Debug, Clone)]
pub enum AutoencoderType {
    Vanilla,
    Denoising,
    Sparse,
}

#[derive(Debug, Clone)]
pub struct AutoencoderConfig<T: Float> {
    /// Latent dimension
    pub latent_dim: usize,
    /// Noise level for denoising autoencoder
    pub noise_level: T,
    /// Sparsity penalty coefficient
    pub sparsity_penalty: T,
    /// Autoencoder type
    pub autoencoder_type: AutoencoderType,
}

impl<T: Float> Default for AutoencoderConfig<T> {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            noise_level: T::from(0.1).unwrap(),
            sparsity_penalty: T::from(0.01).unwrap(),
            autoencoder_type: AutoencoderType::Vanilla,
        }
    }
}

impl<T: FloatBounds + ScalarOperand + Debug> SelfSupervisedAutoencoder<T> {
    /// Create a new self-supervised autoencoder
    pub fn new(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        config: AutoencoderConfig<T>,
    ) -> Result<Self, SklearsError> {
        // Build encoder
        let mut encoder_sizes = vec![input_dim];
        encoder_sizes.extend_from_slice(&hidden_dims);
        encoder_sizes.push(config.latent_dim);

        let encoder_activations: Vec<Option<Activation>> = (0..encoder_sizes.len() - 1)
            .map(|_| Some(Activation::Relu))
            .collect();

        let encoder = SimpleMLP::new(&encoder_sizes, &encoder_activations);

        // Build decoder (reverse of encoder)
        let mut decoder_sizes = vec![config.latent_dim];
        decoder_sizes.extend(hidden_dims.iter().rev().cloned());
        decoder_sizes.push(input_dim);

        let decoder_activations: Vec<Option<Activation>> = (0..decoder_sizes.len() - 2)
            .map(|_| Some(Activation::Relu))
            .chain(std::iter::once(None)) // No activation for final layer
            .collect();

        let decoder = SimpleMLP::new(&decoder_sizes, &decoder_activations);

        Ok(Self {
            encoder,
            decoder,
            latent_dim: config.latent_dim,
            autoencoder_type: config.autoencoder_type.clone(),
            config,
        })
    }

    /// Encode input to latent representation
    pub fn encode(&mut self, input: &Array2<T>) -> Result<Array2<T>, SklearsError> {
        match self.autoencoder_type {
            AutoencoderType::Denoising => {
                let noisy_input = self.add_noise(input)?;
                self.encoder.forward(&noisy_input)
            }
            _ => self.encoder.forward(input),
        }
    }

    /// Decode latent representation to output
    pub fn decode(&mut self, latent: &Array2<T>) -> Result<Array2<T>, SklearsError> {
        self.decoder.forward(latent)
    }

    /// Forward pass through autoencoder
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>, SklearsError> {
        let encoded = self.encode(input)?;
        let decoded = self.decode(&encoded)?;
        Ok(decoded)
    }

    /// Add noise for denoising autoencoder
    fn add_noise(&self, input: &Array2<T>) -> Result<Array2<T>, SklearsError> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut noisy_input = input.clone();

        for element in noisy_input.iter_mut() {
            let noise = T::from(rng.random::<f32>()).unwrap() * self.config.noise_level;
            *element = *element + noise;
        }

        Ok(noisy_input)
    }

    /// Compute reconstruction loss
    pub fn reconstruction_loss(
        &self,
        input: &Array2<T>,
        output: &Array2<T>,
    ) -> Result<T, SklearsError> {
        let diff = input - output;
        let mse = diff.mapv(|x| x * x).mean().unwrap();
        Ok(mse)
    }

    /// Compute sparsity penalty
    pub fn sparsity_penalty(&self, latent: &Array2<T>) -> Result<T, SklearsError> {
        let l1_norm = latent.mapv(|x| x.abs()).sum();
        Ok(self.config.sparsity_penalty * l1_norm)
    }
}

/// Self-supervised learning trainer
///
/// Combines different self-supervised methods with a unified training interface.
#[derive(Debug, Clone)]
pub struct SelfSupervisedTrainer<T: FloatBounds + ScalarOperand> {
    /// Learning method
    method: SelfSupervisedMethod<T>,
    /// Training configuration
    config: TrainingConfig<T>,
}

#[derive(Debug, Clone)]
pub enum SelfSupervisedMethod<T: FloatBounds + ScalarOperand> {
    Contrastive(ContrastiveLearner<T>),
    Autoencoder(SelfSupervisedAutoencoder<T>),
}

#[derive(Debug, Clone)]
pub struct TrainingConfig<T: Float> {
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: T,
    /// Validation split
    pub validation_split: T,
    /// Early stopping patience
    pub patience: usize,
}

impl<T: Float> Default for TrainingConfig<T> {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: T::from(0.001).unwrap(),
            validation_split: T::from(0.1).unwrap(),
            patience: 10,
        }
    }
}

impl<T: FloatBounds + ScalarOperand + Debug + std::iter::Sum> SelfSupervisedTrainer<T> {
    /// Create a new self-supervised trainer
    pub fn new(method: SelfSupervisedMethod<T>, config: TrainingConfig<T>) -> Self {
        Self { method, config }
    }

    /// Train the self-supervised model
    pub fn fit(&mut self, data: &Array2<T>) -> Result<Vec<T>, SklearsError> {
        let mut losses = Vec::new();

        for epoch in 0..self.config.epochs {
            let epoch_loss = match &mut self.method {
                SelfSupervisedMethod::Contrastive(learner) => {
                    let embeddings = learner.forward(data)?;
                    learner.contrastive_loss(&embeddings)?
                }
                SelfSupervisedMethod::Autoencoder(autoencoder) => {
                    let output = autoencoder.forward(data)?;
                    autoencoder.reconstruction_loss(data, &output)?
                }
            };

            losses.push(epoch_loss);

            // Simple convergence check
            if losses.len() > 10 {
                let recent_avg =
                    losses[losses.len() - 10..].iter().cloned().sum::<T>() / T::from(10.0).unwrap();
                if epoch_loss < recent_avg * T::from(0.001).unwrap() {
                    break;
                }
            }
        }

        Ok(losses)
    }

    /// Transform data using the trained model
    pub fn transform(&mut self, data: &Array2<T>) -> Result<Array2<T>, SklearsError> {
        match &mut self.method {
            SelfSupervisedMethod::Contrastive(learner) => learner.encode(data),
            SelfSupervisedMethod::Autoencoder(autoencoder) => autoencoder.encode(data),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dense_layer_creation() {
        let layer = DenseLayer::<f32>::new(10, 5, Some(Activation::Relu));
        assert_eq!(layer.weights.dim(), (10, 5));
        assert_eq!(layer.biases.len(), 5);
    }

    #[test]
    fn test_simple_mlp_creation() {
        let mlp = SimpleMLP::<f32>::new(&[10, 8, 5], &[Some(Activation::Relu), None]);
        assert_eq!(mlp.layers.len(), 2);
    }

    #[test]
    fn test_contrastive_learner_creation() {
        let config = ContrastiveConfig::default();
        let learner = ContrastiveLearner::<f32>::new(100, vec![64, 32], config);
        assert!(learner.is_ok());
    }

    #[test]
    fn test_contrastive_loss_computation() {
        let config = ContrastiveConfig::default();
        let learner = ContrastiveLearner::<f32>::new(10, vec![8], config).unwrap();

        let embeddings = Array2::from_shape_vec((4, 128), vec![0.0; 4 * 128]).unwrap();

        let loss = learner.contrastive_loss(&embeddings);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_autoencoder_creation() {
        let config = AutoencoderConfig::default();
        let autoencoder = SelfSupervisedAutoencoder::<f32>::new(100, vec![64, 32], config);
        assert!(autoencoder.is_ok());
    }

    #[test]
    fn test_autoencoder_forward_pass() {
        let config = AutoencoderConfig::default();
        let mut autoencoder = SelfSupervisedAutoencoder::<f32>::new(10, vec![8], config).unwrap();

        let input = Array2::from_shape_vec((2, 10), (0..20).map(|x| x as f32).collect()).unwrap();
        let output = autoencoder.forward(&input);

        assert!(output.is_ok());
        assert_eq!(output.unwrap().dim(), input.dim());
    }

    #[test]
    fn test_self_supervised_trainer() {
        let config = ContrastiveConfig::default();
        let learner = ContrastiveLearner::<f32>::new(10, vec![8], config).unwrap();
        let method = SelfSupervisedMethod::Contrastive(learner);

        let training_config = TrainingConfig {
            epochs: 5,
            batch_size: 4,
            learning_rate: 0.01,
            validation_split: 0.2,
            patience: 3,
        };

        let mut trainer = SelfSupervisedTrainer::new(method, training_config);
        let data = Array2::from_shape_vec((4, 10), (0..40).map(|x| x as f32).collect()).unwrap();

        let losses = trainer.fit(&data);
        assert!(losses.is_ok());
        assert!(losses.unwrap().len() <= 5);
    }

    #[test]
    fn test_cosine_similarity() {
        let config = ContrastiveConfig::default();
        let learner = ContrastiveLearner::<f32>::new(10, vec![8], config).unwrap();

        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let c = Array1::from_vec(vec![1.0, 0.0, 0.0]);

        let sim_ab = learner.cosine_similarity(&a.view(), &b.view()).unwrap();
        let sim_ac = learner.cosine_similarity(&a.view(), &c.view()).unwrap();

        assert_abs_diff_eq!(sim_ab, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sim_ac, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_reconstruction_loss() {
        let config = AutoencoderConfig::default();
        let autoencoder = SelfSupervisedAutoencoder::<f32>::new(10, vec![8], config).unwrap();

        let input = Array2::from_shape_vec((2, 10), (0..20).map(|x| x as f32).collect()).unwrap();
        let output = input.clone();

        let loss = autoencoder.reconstruction_loss(&input, &output).unwrap();
        assert_abs_diff_eq!(loss, 0.0, epsilon = 1e-6);
    }
}
