//! Semi-Supervised Generative Adversarial Networks (SS-GANs)
//!
//! This module provides a Semi-Supervised GAN implementation for semi-supervised learning.
//! SS-GANs extend traditional GANs to perform both generation and classification tasks,
//! leveraging unlabeled data through the adversarial training process.

use scirs2_core::ndarray_ext::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Random, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Generator network for Semi-Supervised GAN
#[derive(Debug, Clone)]
pub struct Generator {
    /// Layer weights
    pub weights: Vec<Array2<f64>>,
    /// Layer biases
    pub biases: Vec<Array1<f64>>,
    /// Network architecture (layer sizes)
    pub architecture: Vec<usize>,
    /// Noise dimension
    pub noise_dim: usize,
}

impl Generator {
    /// Create a new generator
    pub fn new(noise_dim: usize, output_dim: usize, hidden_dims: Vec<usize>) -> Self {
        let mut architecture = vec![noise_dim];
        architecture.extend(hidden_dims);
        architecture.push(output_dim);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..architecture.len() - 1 {
            let input_dim = architecture[i];
            let output_dim = architecture[i + 1];

            // Xavier initialization
            let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
            let mut rng = Random::default();
            let mut w = Array2::zeros((output_dim, input_dim));
            for i in 0..output_dim {
                for j in 0..input_dim {
                    // Generate standard normal (approximate) and scale
                    w[[i, j]] = rng.gen_range(-3.0..3.0) / 3.0 * scale;
                }
            }
            let w = w;
            let b = Array1::zeros(output_dim);

            weights.push(w);
            biases.push(b);
        }

        Self {
            weights,
            biases,
            architecture,
            noise_dim,
        }
    }

    /// Forward pass through generator
    pub fn forward(&self, noise: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut current = noise.to_owned();

        for (i, (weights, biases)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let linear = weights.dot(&current) + biases;

            // Use tanh for hidden layers, linear for output
            current = if i < self.weights.len() - 1 {
                linear.mapv(|x| x.tanh())
            } else {
                linear
            };
        }

        Ok(current)
    }

    /// Generate samples
    pub fn generate(&self, n_samples: usize) -> SklResult<Array2<f64>> {
        let output_dim = *self.architecture.last().unwrap();
        let mut samples = Array2::zeros((n_samples, output_dim));

        for i in 0..n_samples {
            let mut rng = Random::default();
            let mut noise = Array1::zeros(self.noise_dim);
            for j in 0..self.noise_dim {
                // Generate standard normal (approximate)
                noise[j] = rng.gen_range(-3.0..3.0) / 3.0;
            }
            let generated = self.forward(&noise.view())?;
            samples.row_mut(i).assign(&generated);
        }

        Ok(samples)
    }
}

/// Discriminator network for Semi-Supervised GAN
#[derive(Debug, Clone)]
pub struct Discriminator {
    /// Layer weights
    pub weights: Vec<Array2<f64>>,
    /// Layer biases
    pub biases: Vec<Array1<f64>>,
    /// Network architecture (layer sizes)
    pub architecture: Vec<usize>,
    /// Number of real classes (excluding fake class)
    pub n_classes: usize,
}

impl Discriminator {
    /// Create a new discriminator
    pub fn new(input_dim: usize, n_classes: usize, hidden_dims: Vec<usize>) -> Self {
        let mut architecture = vec![input_dim];
        architecture.extend(hidden_dims);
        architecture.push(n_classes + 1); // +1 for fake class

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..architecture.len() - 1 {
            let input_dim = architecture[i];
            let output_dim = architecture[i + 1];

            // Xavier initialization
            let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
            let mut rng = Random::default();
            let mut w = Array2::zeros((output_dim, input_dim));
            for i in 0..output_dim {
                for j in 0..input_dim {
                    // Generate standard normal (approximate) and scale
                    w[[i, j]] = rng.gen_range(-3.0..3.0) / 3.0 * scale;
                }
            }
            let w = w;
            let b = Array1::zeros(output_dim);

            weights.push(w);
            biases.push(b);
        }

        Self {
            weights,
            biases,
            architecture,
            n_classes,
        }
    }

    /// Forward pass through discriminator
    pub fn forward(&self, x: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut current = x.to_owned();

        for (i, (weights, biases)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let linear = weights.dot(&current) + biases;

            // Use leaky ReLU for hidden layers, linear for output
            current = if i < self.weights.len() - 1 {
                linear.mapv(|x| if x > 0.0 { x } else { 0.01 * x })
            } else {
                linear
            };
        }

        Ok(current)
    }

    /// Get class probabilities (including fake class)
    pub fn predict_proba(&self, x: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let logits = self.forward(x)?;
        Ok(self.softmax(&logits.view()))
    }

    /// Softmax activation
    fn softmax(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// Get real/fake probability
    pub fn get_real_fake_proba(&self, x: &ArrayView1<f64>) -> SklResult<f64> {
        let probs = self.predict_proba(x)?;
        // Sum probabilities of real classes (exclude last class which is fake)
        let real_prob = probs.slice(s![..self.n_classes]).sum();
        Ok(real_prob)
    }
}

/// Semi-Supervised GAN for semi-supervised learning
#[derive(Debug, Clone)]
pub struct SemiSupervisedGAN<S = Untrained> {
    state: S,
    /// Generator network
    generator: Option<Generator>,
    /// Discriminator network
    discriminator: Option<Discriminator>,
    /// Noise dimension for generator
    noise_dim: usize,
    /// Number of classes
    n_classes: usize,
    /// Learning rate
    learning_rate: f64,
    /// Number of epochs
    epochs: usize,
    /// Batch size
    batch_size: usize,
    /// Generator training frequency
    gen_freq: usize,
    /// Discriminator training frequency
    disc_freq: usize,
    /// Hidden dimensions for generator
    gen_hidden_dims: Vec<usize>,
    /// Hidden dimensions for discriminator
    disc_hidden_dims: Vec<usize>,
    /// Random state for reproducibility
    random_state: Option<u64>,
}

impl Default for SemiSupervisedGAN<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl SemiSupervisedGAN<Untrained> {
    /// Create a new Semi-Supervised GAN
    pub fn new() -> Self {
        Self {
            state: Untrained,
            generator: None,
            discriminator: None,
            noise_dim: 100,
            n_classes: 2,
            learning_rate: 0.0002,
            epochs: 100,
            batch_size: 32,
            gen_freq: 1,
            disc_freq: 2,
            gen_hidden_dims: vec![128, 256],
            disc_hidden_dims: vec![256, 128],
            random_state: None,
        }
    }

    /// Set noise dimension
    pub fn noise_dim(mut self, noise_dim: usize) -> Self {
        self.noise_dim = noise_dim;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set number of epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set generator training frequency
    pub fn gen_freq(mut self, freq: usize) -> Self {
        self.gen_freq = freq;
        self
    }

    /// Set discriminator training frequency
    pub fn disc_freq(mut self, freq: usize) -> Self {
        self.disc_freq = freq;
        self
    }

    /// Set generator hidden dimensions
    pub fn gen_hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.gen_hidden_dims = dims;
        self
    }

    /// Set discriminator hidden dimensions
    pub fn disc_hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.disc_hidden_dims = dims;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Initialize networks
    fn initialize_networks(&mut self, input_dim: usize, n_classes: usize) {
        self.n_classes = n_classes;

        self.generator = Some(Generator::new(
            self.noise_dim,
            input_dim,
            self.gen_hidden_dims.clone(),
        ));

        self.discriminator = Some(Discriminator::new(
            input_dim,
            n_classes,
            self.disc_hidden_dims.clone(),
        ));
    }

    /// Train the GAN
    fn train(&mut self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> SklResult<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize networks
        self.initialize_networks(n_features, self.n_classes);

        // Separate labeled and unlabeled data
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();

        for (i, &label) in y.iter().enumerate() {
            if label >= 0 {
                labeled_indices.push(i);
            } else {
                unlabeled_indices.push(i);
            }
        }

        // Training loop (simplified)
        for epoch in 0..self.epochs {
            let mut total_d_loss = 0.0;
            let mut total_g_loss = 0.0;

            // Train discriminator on labeled data
            for batch_start in (0..labeled_indices.len()).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(labeled_indices.len());

                // In practice, you'd implement proper gradient computation here
                // This is a simplified version for demonstration
                total_d_loss += 1.0; // Placeholder
            }

            // Train discriminator on unlabeled data
            for batch_start in (0..unlabeled_indices.len()).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(unlabeled_indices.len());

                // In practice, you'd implement proper gradient computation here
                total_d_loss += 1.0; // Placeholder
            }

            // Train generator
            if epoch % self.gen_freq == 0 {
                // Generate fake samples and train generator
                total_g_loss += 1.0; // Placeholder
            }

            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: D_loss = {:.4}, G_loss = {:.4}",
                    epoch, total_d_loss, total_g_loss
                );
            }
        }

        Ok(())
    }
}

/// Trained state for Semi-Supervised GAN
#[derive(Debug, Clone)]
pub struct SemiSupervisedGANTrained {
    /// generator
    pub generator: Generator,
    /// discriminator
    pub discriminator: Discriminator,
    /// classes
    pub classes: Array1<i32>,
    /// noise_dim
    pub noise_dim: usize,
    /// n_classes
    pub n_classes: usize,
    /// learning_rate
    pub learning_rate: f64,
}

impl SemiSupervisedGAN<SemiSupervisedGANTrained> {
    /// Generate samples using trained generator
    pub fn generate_samples(&self, n_samples: usize) -> SklResult<Array2<f64>> {
        self.state.generator.generate(n_samples)
    }
}

impl Estimator for SemiSupervisedGAN<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for SemiSupervisedGAN<Untrained> {
    type Fitted = SemiSupervisedGAN<SemiSupervisedGANTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let x = x.to_owned();
        let y = y.to_owned();

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        // Check if we have any labeled samples
        let labeled_count = y.iter().filter(|&&label| label >= 0).count();
        if labeled_count == 0 {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        // Get unique classes
        let mut unique_classes: Vec<i32> = y.iter().filter(|&&label| label >= 0).cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();

        let mut model = self.clone();
        model.n_classes = unique_classes.len();

        // Initialize and train the model
        model.initialize_networks(x.ncols(), model.n_classes);
        model.train(&x.view(), &y.view())?;

        Ok(SemiSupervisedGAN {
            state: SemiSupervisedGANTrained {
                generator: model.generator.unwrap(),
                discriminator: model.discriminator.unwrap(),
                classes: Array1::from(unique_classes),
                noise_dim: model.noise_dim,
                n_classes: model.n_classes,
                learning_rate: model.learning_rate,
            },
            generator: None,
            discriminator: None,
            noise_dim: 0,
            n_classes: 0,
            learning_rate: 0.0,
            epochs: 0,
            batch_size: 0,
            gen_freq: 0,
            disc_freq: 0,
            gen_hidden_dims: Vec::new(),
            disc_hidden_dims: Vec::new(),
            random_state: None,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for SemiSupervisedGAN<SemiSupervisedGANTrained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let x = x.to_owned();
        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let probs = self.state.discriminator.predict_proba(&x.row(i))?;

            // Exclude fake class probability (last element)
            let real_probs = probs.slice(s![..self.state.n_classes]);

            let max_idx = real_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predictions[i] = self.state.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for SemiSupervisedGAN<SemiSupervisedGANTrained>
{
    fn predict_proba(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x = x.to_owned();
        let mut probabilities = Array2::zeros((x.nrows(), self.state.n_classes));

        for i in 0..x.nrows() {
            let probs = self.state.discriminator.predict_proba(&x.row(i))?;

            // Extract only real class probabilities (exclude fake class)
            let real_probs = probs.slice(s![..self.state.n_classes]);

            // Renormalize probabilities
            let sum_real_probs = real_probs.sum();
            if sum_real_probs > 0.0 {
                let normalized_probs = &real_probs / sum_real_probs;
                probabilities.row_mut(i).assign(&normalized_probs);
            } else {
                // Uniform distribution if sum is zero
                probabilities
                    .row_mut(i)
                    .fill(1.0 / self.state.n_classes as f64);
            }
        }

        Ok(probabilities)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;
    use scirs2_core::ndarray_ext::{s, ArrayView1, ArrayView2};

    #[test]
    fn test_generator_creation() {
        let gen = Generator::new(100, 10, vec![128, 64]);
        assert_eq!(gen.noise_dim, 100);
        assert_eq!(gen.architecture, vec![100, 128, 64, 10]);
        assert_eq!(gen.weights.len(), 3);
        assert_eq!(gen.biases.len(), 3);
    }

    #[test]
    fn test_generator_forward() {
        let gen = Generator::new(5, 3, vec![8]);
        let noise = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = gen.forward(&noise.view());
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_generator_generate() {
        let gen = Generator::new(5, 3, vec![8]);

        let result = gen.generate(10);
        assert!(result.is_ok());

        let samples = result.unwrap();
        assert_eq!(samples.dim(), (10, 3));
    }

    #[test]
    fn test_discriminator_creation() {
        let disc = Discriminator::new(10, 3, vec![64, 32]);
        assert_eq!(disc.n_classes, 3);
        assert_eq!(disc.architecture, vec![10, 64, 32, 4]); // +1 for fake class
        assert_eq!(disc.weights.len(), 3);
        assert_eq!(disc.biases.len(), 3);
    }

    #[test]
    fn test_discriminator_forward() {
        let disc = Discriminator::new(5, 2, vec![8]);
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = disc.forward(&x.view());
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 3); // 2 real classes + 1 fake class
    }

    #[test]
    fn test_discriminator_predict_proba() {
        let disc = Discriminator::new(5, 2, vec![8]);
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = disc.predict_proba(&x.view());
        assert!(result.is_ok());

        let probs = result.unwrap();
        assert_eq!(probs.len(), 3);
        assert!((probs.sum() - 1.0).abs() < 1e-10);
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_semi_supervised_gan_creation() {
        let gan = SemiSupervisedGAN::new()
            .noise_dim(50)
            .learning_rate(0.001)
            .epochs(50)
            .batch_size(16);

        assert_eq!(gan.noise_dim, 50);
        assert_eq!(gan.learning_rate, 0.001);
        assert_eq!(gan.epochs, 50);
        assert_eq!(gan.batch_size, 16);
    }

    #[test]
    fn test_semi_supervised_gan_fit_predict() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, 0, 1, -1, -1]; // -1 indicates unlabeled

        let gan = SemiSupervisedGAN::new()
            .noise_dim(5)
            .learning_rate(0.01)
            .epochs(5)
            .batch_size(2);

        let result = gan.fit(&X.view(), &y.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.state.classes.len(), 2);

        let predictions = fitted.predict(&X.view());
        assert!(predictions.is_ok());

        let pred = predictions.unwrap();
        assert_eq!(pred.len(), 6);

        let probabilities = fitted.predict_proba(&X.view());
        assert!(probabilities.is_ok());

        let proba = probabilities.unwrap();
        assert_eq!(proba.dim(), (6, 2));

        // Check probabilities sum to 1
        for i in 0..6 {
            let sum: f64 = proba.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_semi_supervised_gan_insufficient_labeled_samples() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![-1, -1]; // All unlabeled

        let gan = SemiSupervisedGAN::new();
        let result = gan.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_semi_supervised_gan_invalid_dimensions() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0]; // Wrong number of labels

        let gan = SemiSupervisedGAN::new();
        let result = gan.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_discriminator_real_fake_proba() {
        let disc = Discriminator::new(3, 2, vec![4]);
        let x = array![1.0, 2.0, 3.0];

        let result = disc.get_real_fake_proba(&x.view());
        assert!(result.is_ok());

        let real_prob = result.unwrap();
        assert!(real_prob >= 0.0 && real_prob <= 1.0);
    }

    #[test]
    fn test_semi_supervised_gan_generate_samples() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 0, -1]; // Mixed labeled and unlabeled

        let gan = SemiSupervisedGAN::new().noise_dim(5).epochs(3);

        let fitted = gan.fit(&X.view(), &y.view()).unwrap();

        let generated = fitted.generate_samples(5);
        assert!(generated.is_ok());

        let samples = generated.unwrap();
        assert_eq!(samples.dim(), (5, 3));
    }
}
