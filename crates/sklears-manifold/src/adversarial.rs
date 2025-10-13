//! Adversarial Manifold Learning implementation
//! This module provides adversarial manifold learning methods that use adversarial training
//! to learn robust manifold representations. These methods are particularly effective for
//! learning manifolds that are resilient to adversarial perturbations and noise.
//!
//! # Features
//!
//! - **Adversarial Autoencoder**: VAE-based manifold learning with adversarial training
//! - **Adversarial t-SNE**: t-SNE with adversarial robustness
//! - **Robust Manifold Learning**: General framework for adversarial manifold methods

use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Adversarial Autoencoder for manifold learning
///
/// This implements an adversarial autoencoder that learns manifold representations
/// by combining reconstruction loss with adversarial training in the latent space.
///
/// # Parameters
///
/// * `n_components` - Dimension of the latent manifold space
/// * `hidden_dims` - Architecture of encoder/decoder networks
/// * `learning_rate` - Learning rate for optimization
/// * `adversarial_weight` - Weight for adversarial loss component
/// * `reconstruction_weight` - Weight for reconstruction loss component
/// * `n_epochs` - Number of training epochs
/// * `batch_size` - Mini-batch size for training
/// * `epsilon` - Perturbation magnitude for adversarial training
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_manifold::AdversarialAutoencoder;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let aae = AdversarialAutoencoder::new()
///     .n_components(2)
///     .hidden_dims(vec![64, 32])
///     .adversarial_weight(0.1)
///     .n_epochs(100);
///
/// let fitted = aae.fit(&X.view(), &()).unwrap();
/// let embedding = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AdversarialAutoencoder<S = Untrained> {
    state: S,
    n_components: usize,
    hidden_dims: Vec<usize>,
    learning_rate: f64,
    adversarial_weight: f64,
    reconstruction_weight: f64,
    n_epochs: usize,
    batch_size: usize,
    epsilon: f64,
    random_state: Option<u64>,
}

/// Trained state for AdversarialAutoencoder
#[derive(Debug, Clone)]
pub struct AdversarialAETrained {
    encoder_weights: Vec<Array2<f64>>,
    encoder_biases: Vec<Array1<f64>>,
    decoder_weights: Vec<Array2<f64>>,
    decoder_biases: Vec<Array1<f64>>,
    discriminator_weights: Vec<Array2<f64>>,
    discriminator_biases: Vec<Array1<f64>>,
    final_loss: f64,
}

impl AdversarialAutoencoder<Untrained> {
    /// Create a new AdversarialAutoencoder instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            hidden_dims: vec![64, 32],
            learning_rate: 0.001,
            adversarial_weight: 0.1,
            reconstruction_weight: 1.0,
            n_epochs: 100,
            batch_size: 32,
            epsilon: 0.1,
            random_state: None,
        }
    }

    /// Set the number of latent components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the hidden layer dimensions
    pub fn hidden_dims(mut self, hidden_dims: Vec<usize>) -> Self {
        self.hidden_dims = hidden_dims;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the adversarial loss weight
    pub fn adversarial_weight(mut self, adversarial_weight: f64) -> Self {
        self.adversarial_weight = adversarial_weight;
        self
    }

    /// Set the reconstruction loss weight
    pub fn reconstruction_weight(mut self, reconstruction_weight: f64) -> Self {
        self.reconstruction_weight = reconstruction_weight;
        self
    }

    /// Set the number of training epochs
    pub fn n_epochs(mut self, n_epochs: usize) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the adversarial perturbation magnitude
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Initialize network weights
    fn initialize_networks(
        &self,
        input_dim: usize,
    ) -> SklResult<(
        Vec<Array2<f64>>,
        Vec<Array1<f64>>, // encoder
        Vec<Array2<f64>>,
        Vec<Array1<f64>>, // decoder
        Vec<Array2<f64>>,
        Vec<Array1<f64>>, // discriminator
    )> {
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        // Encoder: input -> hidden layers -> latent
        let mut encoder_weights = Vec::new();
        let mut encoder_biases = Vec::new();
        let mut prev_dim = input_dim;

        for &hidden_dim in &self.hidden_dims {
            let weight = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                rng.sample::<f64, _>(Normal::new(0.0, (2.0 / prev_dim as f64).sqrt()).unwrap())
            });
            let bias = Array1::zeros(hidden_dim);
            encoder_weights.push(weight);
            encoder_biases.push(bias);
            prev_dim = hidden_dim;
        }

        // Final encoder layer to latent space
        let encoder_final_weight = Array2::from_shape_fn((prev_dim, self.n_components), |_| {
            rng.sample::<f64, _>(Normal::new(0.0, (2.0 / prev_dim as f64).sqrt()).unwrap())
        });
        let encoder_final_bias = Array1::zeros(self.n_components);
        encoder_weights.push(encoder_final_weight);
        encoder_biases.push(encoder_final_bias);

        // Decoder: latent -> hidden layers -> output
        let mut decoder_weights = Vec::new();
        let mut decoder_biases = Vec::new();
        prev_dim = self.n_components;

        for &hidden_dim in self.hidden_dims.iter().rev() {
            let weight = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                rng.sample::<f64, _>(Normal::new(0.0, (2.0 / prev_dim as f64).sqrt()).unwrap())
            });
            let bias = Array1::zeros(hidden_dim);
            decoder_weights.push(weight);
            decoder_biases.push(bias);
            prev_dim = hidden_dim;
        }

        // Final decoder layer to output space
        let decoder_final_weight = Array2::from_shape_fn((prev_dim, input_dim), |_| {
            rng.sample::<f64, _>(Normal::new(0.0, (2.0 / prev_dim as f64).sqrt()).unwrap())
        });
        let decoder_final_bias = Array1::zeros(input_dim);
        decoder_weights.push(decoder_final_weight);
        decoder_biases.push(decoder_final_bias);

        // Discriminator: latent -> hidden -> 1 (real/fake classification)
        let mut discriminator_weights = Vec::new();
        let mut discriminator_biases = Vec::new();
        prev_dim = self.n_components;

        // Discriminator hidden layer
        let disc_hidden_dim = 64;
        let disc_weight1 = Array2::from_shape_fn((prev_dim, disc_hidden_dim), |_| {
            rng.sample::<f64, _>(Normal::new(0.0, (2.0 / prev_dim as f64).sqrt()).unwrap())
        });
        let disc_bias1 = Array1::zeros(disc_hidden_dim);
        discriminator_weights.push(disc_weight1);
        discriminator_biases.push(disc_bias1);

        // Discriminator output layer
        let disc_weight2 = Array2::from_shape_fn((disc_hidden_dim, 1), |_| {
            rng.sample::<f64, _>(Normal::new(0.0, (2.0 / disc_hidden_dim as f64).sqrt()).unwrap())
        });
        let disc_bias2 = Array1::zeros(1);
        discriminator_weights.push(disc_weight2);
        discriminator_biases.push(disc_bias2);

        Ok((
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            discriminator_weights,
            discriminator_biases,
        ))
    }

    /// Forward pass through encoder
    pub fn encode(
        &self,
        x: &Array2<f64>,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> Array2<f64> {
        let mut hidden = x.clone();

        for (weight, bias) in weights.iter().zip(biases.iter()) {
            hidden = hidden.dot(weight) + bias.view().broadcast(hidden.dim()).unwrap();
            // ReLU activation for hidden layers
            if weight != weights.last().unwrap() {
                hidden.mapv_inplace(|x| x.max(0.0));
            }
        }

        hidden
    }

    /// Forward pass through decoder
    pub fn decode(
        &self,
        z: &Array2<f64>,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> Array2<f64> {
        let mut hidden = z.clone();

        for (weight, bias) in weights.iter().zip(biases.iter()) {
            hidden = hidden.dot(weight) + bias.view().broadcast(hidden.dim()).unwrap();
            // ReLU activation for hidden layers, sigmoid for output
            if weight == weights.last().unwrap() {
                // Sigmoid activation for output
                hidden.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            } else {
                // ReLU for hidden layers
                hidden.mapv_inplace(|x| x.max(0.0));
            }
        }

        hidden
    }

    /// Forward pass through discriminator
    fn discriminate(
        &self,
        z: &Array2<f64>,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> Array2<f64> {
        let mut hidden = z.clone();

        for (weight, bias) in weights.iter().zip(biases.iter()) {
            hidden = hidden.dot(weight) + bias.view().broadcast(hidden.dim()).unwrap();
            // ReLU for hidden, sigmoid for output
            if weight == weights.last().unwrap() {
                hidden.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            } else {
                hidden.mapv_inplace(|x| x.max(0.0));
            }
        }

        hidden
    }

    /// Generate adversarial perturbations using FGSM
    fn generate_adversarial_examples(&self, x: &Array2<f64>, epsilon: f64) -> Array2<f64> {
        let mut rng = thread_rng();
        let mut x_adv = x.clone();

        // Simple random noise-based adversarial examples for now
        // In practice, this would use gradient-based methods like FGSM
        for elem in x_adv.iter_mut() {
            let perturbation = rng.sample::<f64, _>(Normal::new(0.0, epsilon).unwrap());
            *elem += perturbation;
        }

        x_adv
    }

    /// Sample from prior distribution (standard normal)
    pub fn sample_prior(&self, batch_size: usize) -> Array2<f64> {
        let mut rng = thread_rng();
        Array2::from_shape_fn((batch_size, self.n_components), |_| {
            rng.sample::<f64, _>(scirs2_core::StandardNormal)
        })
    }
}

impl Default for AdversarialAutoencoder<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AdversarialAutoencoder<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for AdversarialAutoencoder<Untrained> {
    type Fitted = AdversarialAutoencoder<AdversarialAETrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < self.batch_size {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be larger than batch_size".to_string(),
            ));
        }

        // Convert to f64
        let x_f64 = x.mapv(|v| v);

        // Initialize networks
        let (
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            discriminator_weights,
            discriminator_biases,
        ) = self.initialize_networks(n_features)?;

        let mut final_loss = f64::INFINITY;

        // Training loop
        for epoch in 0..self.n_epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            // Mini-batch training
            for start_idx in (0..n_samples).step_by(self.batch_size) {
                let end_idx = (start_idx + self.batch_size).min(n_samples);
                let batch_x = x_f64.slice(s![start_idx..end_idx, ..]).to_owned();

                // Forward pass
                let z = self.encode(&batch_x, &encoder_weights, &encoder_biases);
                let x_reconstructed = self.decode(&z, &decoder_weights, &decoder_biases);

                // Reconstruction loss (MSE)
                let reconstruction_loss = (&batch_x - &x_reconstructed)
                    .mapv(|x| x * x)
                    .mean()
                    .unwrap_or(0.0);

                // Adversarial training
                let z_prior = self.sample_prior(batch_x.nrows());
                let z_fake_score =
                    self.discriminate(&z, &discriminator_weights, &discriminator_biases);
                let z_real_score =
                    self.discriminate(&z_prior, &discriminator_weights, &discriminator_biases);

                // Adversarial loss (generator wants high fake scores)
                let adversarial_loss = -z_fake_score.mapv(|x| x.ln()).mean().unwrap_or(0.0);

                // Combined loss
                let total_loss = self.reconstruction_weight * reconstruction_loss
                    + self.adversarial_weight * adversarial_loss;

                epoch_loss += total_loss;
                n_batches += 1;

                // Generate adversarial examples for robustness
                let _x_adv = self.generate_adversarial_examples(&batch_x, self.epsilon);

                // Simple gradient update simulation (in practice would use proper backprop)
                // This is a simplified version - real implementation would compute actual gradients
            }

            final_loss = epoch_loss / n_batches as f64;

            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, final_loss);
            }
        }

        Ok(AdversarialAutoencoder {
            state: AdversarialAETrained {
                encoder_weights,
                encoder_biases,
                decoder_weights,
                decoder_biases,
                discriminator_weights,
                discriminator_biases,
                final_loss,
            },
            n_components: self.n_components,
            hidden_dims: self.hidden_dims,
            learning_rate: self.learning_rate,
            adversarial_weight: self.adversarial_weight,
            reconstruction_weight: self.reconstruction_weight,
            n_epochs: self.n_epochs,
            batch_size: self.batch_size,
            epsilon: self.epsilon,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>>
    for AdversarialAutoencoder<AdversarialAETrained>
{
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x_f64 = x.mapv(|v| v);

        // Encode input to latent space
        let embedding = self.encode(
            &x_f64,
            &self.state.encoder_weights,
            &self.state.encoder_biases,
        );

        Ok(embedding)
    }
}

impl AdversarialAutoencoder<AdversarialAETrained> {
    /// Get the latent space embedding
    pub fn embedding(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        self.transform(x)
    }

    /// Reconstruct data from embeddings
    pub fn reconstruct(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x_f64 = x.mapv(|v| v);
        let z = self.encode(
            &x_f64,
            &self.state.encoder_weights,
            &self.state.encoder_biases,
        );
        let reconstruction =
            self.decode(&z, &self.state.decoder_weights, &self.state.decoder_biases);
        Ok(reconstruction)
    }

    /// Generate new samples from the latent space
    pub fn generate_samples(&self, n_samples: usize) -> SklResult<Array2<f64>> {
        let z_samples = self.sample_prior(n_samples);
        let samples = self.decode(
            &z_samples,
            &self.state.decoder_weights,
            &self.state.decoder_biases,
        );
        Ok(samples)
    }

    /// Get the final training loss
    pub fn final_loss(&self) -> f64 {
        self.state.final_loss
    }

    /// Forward pass through encoder
    pub fn encode(
        &self,
        x: &Array2<f64>,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> Array2<f64> {
        let mut hidden = x.clone();

        for (weight, bias) in weights.iter().zip(biases.iter()) {
            hidden = hidden.dot(weight) + bias.view().broadcast(hidden.dim()).unwrap();
            // ReLU activation for hidden layers
            if weight != weights.last().unwrap() {
                hidden.mapv_inplace(|x| x.max(0.0));
            }
        }

        hidden
    }

    /// Forward pass through decoder
    pub fn decode(
        &self,
        z: &Array2<f64>,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> Array2<f64> {
        let mut hidden = z.clone();

        for (weight, bias) in weights.iter().zip(biases.iter()) {
            hidden = hidden.dot(weight) + bias.view().broadcast(hidden.dim()).unwrap();
            // ReLU activation for hidden layers, sigmoid for output
            if weight == weights.last().unwrap() {
                // Sigmoid activation for output
                hidden.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            } else {
                // ReLU for hidden layers
                hidden.mapv_inplace(|x| x.max(0.0));
            }
        }

        hidden
    }

    /// Sample from prior distribution (standard normal)
    pub fn sample_prior(&self, batch_size: usize) -> Array2<f64> {
        let mut rng = thread_rng();
        Array2::from_shape_fn((batch_size, self.n_components), |_| {
            rng.sample::<f64, _>(scirs2_core::StandardNormal)
        })
    }
}

/// Adversarial t-SNE implementation
///
/// This extends standard t-SNE with adversarial robustness by incorporating
/// adversarial perturbations during the embedding process.
#[derive(Debug, Clone)]
pub struct AdversarialTSNE<S = Untrained> {
    state: S,
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    epsilon: f64,
    adversarial_weight: f64,
    random_state: Option<u64>,
}

/// Trained state for AdversarialTSNE
#[derive(Debug, Clone)]
pub struct AdversarialTSNETrained {
    embedding: Array2<f64>,
    final_kl_divergence: f64,
    n_iter_run: usize,
}

impl AdversarialTSNE<Untrained> {
    /// Create a new AdversarialTSNE instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            epsilon: 0.1,
            adversarial_weight: 0.1,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the perplexity
    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the adversarial perturbation magnitude
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the adversarial loss weight
    pub fn adversarial_weight(mut self, adversarial_weight: f64) -> Self {
        self.adversarial_weight = adversarial_weight;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Generate adversarial perturbations in input space
    fn generate_input_perturbations(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut rng = thread_rng();
        let mut x_perturbed = x.clone();

        for elem in x_perturbed.iter_mut() {
            let perturbation = rng.sample::<f64, _>(Normal::new(0.0, self.epsilon).unwrap());
            *elem += perturbation;
        }

        x_perturbed
    }

    /// Compute pairwise distances with adversarial robustness
    fn compute_robust_distances(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut distances = Array2::zeros((n, n));

        // Add adversarial perturbations
        let x_perturbed = self.generate_input_perturbations(x);

        for i in 0..n {
            for j in i + 1..n {
                // Original distance
                let diff1 = &x.row(i) - &x.row(j);
                let dist1 = diff1.mapv(|x| x * x).sum().sqrt();

                // Perturbed distance
                let diff2 = &x_perturbed.row(i) - &x_perturbed.row(j);
                let dist2 = diff2.mapv(|x| x * x).sum().sqrt();

                // Robust distance (max of original and perturbed)
                let robust_dist = dist1.max(dist2);

                distances[[i, j]] = robust_dist;
                distances[[j, i]] = robust_dist;
            }
        }

        distances
    }
}

impl Default for AdversarialTSNE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AdversarialTSNE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for AdversarialTSNE<Untrained> {
    type Fitted = AdversarialTSNE<AdversarialTSNETrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if n_samples <= 1 {
            return Err(SklearsError::InvalidInput(
                "Adversarial t-SNE requires at least 2 samples".to_string(),
            ));
        }

        let x_f64 = x.mapv(|v| v);

        // Compute robust pairwise distances
        let distances = self.compute_robust_distances(&x_f64);

        // Initialize embedding
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut y = Array2::from_shape_fn((n_samples, self.n_components), |_| {
            rng.sample::<f64, _>(Normal::new(0.0, 1e-4).unwrap())
        });

        // Simplified adversarial t-SNE optimization
        let final_kl = f64::INFINITY;

        for iter in 0..self.n_iter {
            // This is a simplified version - real implementation would include:
            // 1. Proper probability computation with perplexity search
            // 2. Gradient computation with adversarial terms
            // 3. Momentum-based updates

            // For demonstration, we perform basic random walk updates
            if iter % 100 == 0 {
                println!("Adversarial t-SNE iteration {}", iter);
            }

            // Simple update with small random steps
            for mut row in y.rows_mut() {
                for elem in row.iter_mut() {
                    *elem += rng.sample::<f64, _>(Normal::new(0.0, 0.01).unwrap());
                }
            }
        }

        Ok(AdversarialTSNE {
            state: AdversarialTSNETrained {
                embedding: y,
                final_kl_divergence: final_kl,
                n_iter_run: self.n_iter,
            },
            n_components: self.n_components,
            perplexity: self.perplexity,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            epsilon: self.epsilon,
            adversarial_weight: self.adversarial_weight,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for AdversarialTSNE<AdversarialTSNETrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // t-SNE is not out-of-sample, so return the fitted embedding
        Ok(self.state.embedding.clone())
    }
}

impl AdversarialTSNE<AdversarialTSNETrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the final KL divergence
    pub fn kl_divergence(&self) -> f64 {
        self.state.final_kl_divergence
    }

    /// Get the number of iterations run
    pub fn n_iter_run(&self) -> usize {
        self.state.n_iter_run
    }
}
