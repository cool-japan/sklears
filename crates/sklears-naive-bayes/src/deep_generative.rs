//! Deep Generative Models for Naive Bayes
//!
//! This module provides deep generative model extensions to Naive Bayes classifiers,
//! including Variational Autoencoders, Normalizing Flows, and Neural Posterior Estimation.

use crate::neural_naive_bayes::{ActivationFunction, NeuralLayer, NeuralNBError};
// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
// SciRS2 Policy Compliance - Use scirs2-core for random distributions
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::Distribution;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeepGenerativeError {
    #[error("VAE training failed: {0}")]
    VAETrainingFailed(String),
    #[error("Flow inversion failed: {0}")]
    FlowInversionFailed(String),
    #[error("Posterior estimation failed: {0}")]
    PosteriorEstimationFailed(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Invalid latent dimension: {0}")]
    InvalidLatentDimension(usize),
    #[error("Neural network error: {0}")]
    NeuralError(#[from] NeuralNBError),
}

/// Variational Autoencoder for generative Naive Bayes
#[derive(Debug)]
pub struct VariationalAutoencoder {
    encoder_mean: Vec<NeuralLayer>,
    encoder_logvar: Vec<NeuralLayer>,
    decoder: Vec<NeuralLayer>,
    latent_dim: usize,
    input_dim: usize,
    config: VAEConfig,
    rng: scirs2_core::random::CoreRandom<StdRng>,
}

#[derive(Debug, Clone)]
pub struct VAEConfig {
    pub latent_dim: usize,
    pub hidden_layers: Vec<usize>,
    pub learning_rate: f64,
    pub beta: f64, // KL divergence weight
    pub max_epochs: usize,
    pub batch_size: usize,
    pub tolerance: f64,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            latent_dim: 10,
            hidden_layers: vec![128, 64],
            learning_rate: 0.001,
            beta: 1.0,
            max_epochs: 1000,
            batch_size: 32,
            tolerance: 1e-6,
        }
    }
}

impl VariationalAutoencoder {
    pub fn new(input_dim: usize, config: VAEConfig) -> Self {
        let mut rng = scirs2_core::random::CoreRandom::<StdRng>::from_rng(
            &mut scirs2_core::random::thread_rng(),
        );

        Self {
            encoder_mean: Self::build_encoder(&config, input_dim, &mut rng),
            encoder_logvar: Self::build_encoder(&config, input_dim, &mut rng),
            decoder: Self::build_decoder(&config, input_dim, &mut rng),
            latent_dim: config.latent_dim,
            input_dim,
            config,
            rng,
        }
    }

    fn build_encoder(
        config: &VAEConfig,
        input_dim: usize,
        rng: &mut scirs2_core::random::CoreRandom<StdRng>,
    ) -> Vec<NeuralLayer> {
        let mut layers = Vec::new();
        let mut current_size = input_dim;

        // Hidden layers
        for &hidden_size in &config.hidden_layers {
            layers.push(NeuralLayer::new(
                current_size,
                hidden_size,
                ActivationFunction::ReLU,
                rng,
            ));
            current_size = hidden_size;
        }

        // Output layer (mean or logvar)
        layers.push(NeuralLayer::new(
            current_size,
            config.latent_dim,
            ActivationFunction::Identity,
            rng,
        ));

        layers
    }

    fn build_decoder(
        config: &VAEConfig,
        input_dim: usize,
        rng: &mut scirs2_core::random::CoreRandom<StdRng>,
    ) -> Vec<NeuralLayer> {
        let mut layers = Vec::new();
        let mut current_size = config.latent_dim;

        // Hidden layers (reverse of encoder)
        for &hidden_size in config.hidden_layers.iter().rev() {
            layers.push(NeuralLayer::new(
                current_size,
                hidden_size,
                ActivationFunction::ReLU,
                rng,
            ));
            current_size = hidden_size;
        }

        // Output layer
        layers.push(NeuralLayer::new(
            current_size,
            input_dim,
            ActivationFunction::Sigmoid,
            rng,
        ));

        layers
    }

    /// Encode input to latent space (mean and logvar)
    pub fn encode(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let mean = self.forward_network(&self.encoder_mean, x);
        let logvar = self.forward_network(&self.encoder_logvar, x);
        (mean, logvar)
    }

    /// Decode from latent space
    pub fn decode(&self, z: &Array1<f64>) -> Array1<f64> {
        self.forward_network(&self.decoder, z)
    }

    /// Sample from latent distribution using reparameterization trick
    pub fn reparameterize(&mut self, mean: &Array1<f64>, logvar: &Array1<f64>) -> Array1<f64> {
        let std_dev = logvar.mapv(|x| (0.5 * x).exp());
        let normal = RandNormal::new(0.0, 1.0).unwrap();

        let epsilon: Array1<f64> = (0..mean.len())
            .map(|_| normal.sample(&mut self.rng))
            .collect::<Vec<_>>()
            .into();

        mean + &(std_dev * epsilon)
    }

    /// Forward pass through network
    fn forward_network(&self, network: &[NeuralLayer], input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in network {
            output = layer.forward(&output);
        }
        output
    }

    /// Compute VAE loss (reconstruction + KL divergence)
    pub fn compute_loss(
        &self,
        x: &Array1<f64>,
        x_recon: &Array1<f64>,
        mean: &Array1<f64>,
        logvar: &Array1<f64>,
    ) -> f64 {
        // Reconstruction loss (binary cross-entropy)
        let recon_loss: f64 = x
            .iter()
            .zip(x_recon.iter())
            .map(|(&x_true, &x_pred)| -x_true * x_pred.ln() - (1.0 - x_true) * (1.0 - x_pred).ln())
            .sum();

        // KL divergence loss
        let kl_loss: f64 = -0.5
            * mean
                .iter()
                .zip(logvar.iter())
                .map(|(&mu, &logvar)| 1.0 + logvar - mu.powi(2) - logvar.exp())
                .sum::<f64>();

        recon_loss + self.config.beta * kl_loss
    }

    /// Generate new samples
    pub fn generate(&mut self, n_samples: usize) -> Array2<f64> {
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut samples = Array2::zeros((n_samples, self.input_dim));

        for i in 0..n_samples {
            let z: Array1<f64> = (0..self.latent_dim)
                .map(|_| normal.sample(&mut self.rng))
                .collect::<Vec<_>>()
                .into();

            let generated = self.decode(&z);
            samples.row_mut(i).assign(&generated);
        }

        samples
    }
}

/// Normalizing Flow for distribution learning
#[derive(Debug)]
pub struct NormalizingFlow {
    coupling_layers: Vec<CouplingLayer>,
    config: FlowConfig,
    rng: scirs2_core::random::CoreRandom<StdRng>,
}

#[derive(Debug, Clone)]
pub struct FlowConfig {
    pub n_layers: usize,
    pub hidden_units: usize,
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub batch_size: usize,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            n_layers: 8,
            hidden_units: 64,
            learning_rate: 0.001,
            max_epochs: 1000,
            batch_size: 32,
        }
    }
}

#[derive(Debug)]
struct CouplingLayer {
    mask: Array1<bool>,
    scale_net: Vec<NeuralLayer>,
    translate_net: Vec<NeuralLayer>,
}

impl CouplingLayer {
    fn new(
        dim: usize,
        hidden_units: usize,
        mask: Array1<bool>,
        rng: &mut scirs2_core::random::CoreRandom<StdRng>,
    ) -> Self {
        let masked_dim = mask.iter().filter(|&&x| !x).count();
        let output_dim = mask.iter().filter(|&&x| x).count();

        let scale_net = vec![
            NeuralLayer::new(masked_dim, hidden_units, ActivationFunction::ReLU, rng),
            NeuralLayer::new(hidden_units, hidden_units, ActivationFunction::ReLU, rng),
            NeuralLayer::new(hidden_units, output_dim, ActivationFunction::Tanh, rng),
        ];

        let translate_net = vec![
            NeuralLayer::new(masked_dim, hidden_units, ActivationFunction::ReLU, rng),
            NeuralLayer::new(hidden_units, hidden_units, ActivationFunction::ReLU, rng),
            NeuralLayer::new(hidden_units, output_dim, ActivationFunction::Identity, rng),
        ];

        Self {
            mask,
            scale_net,
            translate_net,
        }
    }

    /// Forward transformation
    fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        let masked_x = self.apply_mask(x, false);
        let scale = self.forward_network(&self.scale_net, &masked_x);
        let translate = self.forward_network(&self.translate_net, &masked_x);

        let mut y = x.clone();
        let mut log_det = 0.0;

        let mut scale_idx = 0;
        for (i, &is_transformed) in self.mask.iter().enumerate() {
            if is_transformed {
                y[i] = x[i] * scale[scale_idx].exp() + translate[scale_idx];
                log_det += scale[scale_idx];
                scale_idx += 1;
            }
        }

        (y, log_det)
    }

    /// Inverse transformation
    fn inverse(&self, y: &Array1<f64>) -> (Array1<f64>, f64) {
        let masked_y = self.apply_mask(y, false);
        let scale = self.forward_network(&self.scale_net, &masked_y);
        let translate = self.forward_network(&self.translate_net, &masked_y);

        let mut x = y.clone();
        let mut log_det = 0.0;

        let mut scale_idx = 0;
        for (i, &is_transformed) in self.mask.iter().enumerate() {
            if is_transformed {
                x[i] = (y[i] - translate[scale_idx]) * (-scale[scale_idx]).exp();
                log_det -= scale[scale_idx];
                scale_idx += 1;
            }
        }

        (x, log_det)
    }

    fn apply_mask(&self, x: &Array1<f64>, invert: bool) -> Array1<f64> {
        x.iter()
            .zip(self.mask.iter())
            .filter(|(_, &mask_val)| if invert { mask_val } else { !mask_val })
            .map(|(&val, _)| val)
            .collect::<Vec<_>>()
            .into()
    }

    fn forward_network(&self, network: &[NeuralLayer], input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in network {
            output = layer.forward(&output);
        }
        output
    }
}

impl NormalizingFlow {
    pub fn new(dim: usize, config: FlowConfig) -> Self {
        let mut rng = scirs2_core::random::CoreRandom::<StdRng>::from_rng(
            &mut scirs2_core::random::thread_rng(),
        );
        let mut coupling_layers = Vec::new();

        for i in 0..config.n_layers {
            // Alternate masking pattern
            let mask = Array1::from_shape_fn(dim, |j| (j + i) % 2 == 0);
            coupling_layers.push(CouplingLayer::new(dim, config.hidden_units, mask, &mut rng));
        }

        Self {
            coupling_layers,
            config,
            rng,
        }
    }

    /// Forward pass through the flow
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        let mut z = x.clone();
        let mut log_det_total = 0.0;

        for layer in &self.coupling_layers {
            let (z_new, log_det) = layer.forward(&z);
            z = z_new;
            log_det_total += log_det;
        }

        (z, log_det_total)
    }

    /// Inverse pass through the flow
    pub fn inverse(&self, z: &Array1<f64>) -> (Array1<f64>, f64) {
        let mut x = z.clone();
        let mut log_det_total = 0.0;

        for layer in self.coupling_layers.iter().rev() {
            let (x_new, log_det) = layer.inverse(&x);
            x = x_new;
            log_det_total += log_det;
        }

        (x, log_det_total)
    }

    /// Compute log probability
    pub fn log_prob(&self, x: &Array1<f64>) -> f64 {
        let (z, log_det) = self.forward(x);

        // Standard normal log probability
        let log_prob_z: f64 = z
            .iter()
            .map(|&zi| -0.5 * (zi.powi(2) + (2.0 * std::f64::consts::PI).ln()))
            .sum();

        log_prob_z + log_det
    }

    /// Sample from the flow
    pub fn sample(&mut self, n_samples: usize) -> Array2<f64> {
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let dim = self.coupling_layers[0].mask.len();
        let mut samples = Array2::zeros((n_samples, dim));

        for i in 0..n_samples {
            let z: Array1<f64> = (0..dim)
                .map(|_| normal.sample(&mut self.rng))
                .collect::<Vec<_>>()
                .into();

            let (x, _) = self.inverse(&z);
            samples.row_mut(i).assign(&x);
        }

        samples
    }
}

/// Neural Posterior Estimation
#[derive(Debug)]
pub struct NeuralPosteriorEstimator {
    density_network: Vec<NeuralLayer>,
    config: NPEConfig,
    rng: scirs2_core::random::CoreRandom<StdRng>,
}

#[derive(Debug, Clone)]
pub struct NPEConfig {
    pub hidden_layers: Vec<usize>,
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub batch_size: usize,
    pub n_simulations: usize,
}

impl Default for NPEConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![128, 128, 64],
            learning_rate: 0.001,
            max_epochs: 1000,
            batch_size: 64,
            n_simulations: 10000,
        }
    }
}

impl NeuralPosteriorEstimator {
    pub fn new(param_dim: usize, obs_dim: usize, config: NPEConfig) -> Self {
        let mut rng = scirs2_core::random::CoreRandom::<StdRng>::from_rng(
            &mut scirs2_core::random::thread_rng(),
        );
        let input_dim = param_dim + obs_dim;

        let mut layers = Vec::new();
        let mut current_size = input_dim;

        for &hidden_size in &config.hidden_layers {
            layers.push(NeuralLayer::new(
                current_size,
                hidden_size,
                ActivationFunction::ReLU,
                &mut rng,
            ));
            current_size = hidden_size;
        }

        // Output layer for density estimation
        layers.push(NeuralLayer::new(
            current_size,
            1,
            ActivationFunction::Identity,
            &mut rng,
        ));

        Self {
            density_network: layers,
            config,
            rng,
        }
    }

    /// Estimate log posterior density
    pub fn log_posterior(&self, params: &Array1<f64>, obs: &Array1<f64>) -> f64 {
        let input = self.concatenate_inputs(params, obs);
        let output = self.forward_network(&input);
        output[0]
    }

    /// Sample from posterior using MCMC
    pub fn sample_posterior(&mut self, obs: &Array1<f64>, n_samples: usize) -> Array2<f64> {
        let param_dim = self.density_network[0].input_size() - obs.len();
        let mut samples = Array2::zeros((n_samples, param_dim));

        // Initialize with random sample
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut current_params: Array1<f64> = (0..param_dim)
            .map(|_| normal.sample(&mut self.rng))
            .collect::<Vec<_>>()
            .into();

        let mut current_log_prob = self.log_posterior(&current_params, obs);
        let mut accepted = 0;

        for i in 0..n_samples {
            // Propose new parameters
            let proposal_std = 0.1;
            let proposal: Array1<f64> = current_params
                .iter()
                .map(|&x| {
                    x + RandNormal::new(0.0, proposal_std)
                        .unwrap()
                        .sample(&mut self.rng)
                })
                .collect::<Vec<_>>()
                .into();

            let proposal_log_prob = self.log_posterior(&proposal, obs);

            // Accept/reject
            let log_ratio = proposal_log_prob - current_log_prob;
            let uniform = RandUniform::new(0.0, 1.0).unwrap();
            if log_ratio > 0.0 || uniform.sample(&mut self.rng) < log_ratio.exp() {
                current_params = proposal;
                current_log_prob = proposal_log_prob;
                accepted += 1;
            }

            samples.row_mut(i).assign(&current_params);
        }

        samples
    }

    fn concatenate_inputs(&self, params: &Array1<f64>, obs: &Array1<f64>) -> Array1<f64> {
        let mut input = Array1::zeros(params.len() + obs.len());
        input.slice_mut(s![..params.len()]).assign(params);
        input.slice_mut(s![params.len()..]).assign(obs);
        input
    }

    fn forward_network(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &self.density_network {
            output = layer.forward(&output);
        }
        output
    }
}

/// Deep Generative Naive Bayes combining all methods
#[derive(Debug)]
pub struct DeepGenerativeNaiveBayes {
    vaes: HashMap<i32, VariationalAutoencoder>,
    flows: HashMap<i32, NormalizingFlow>,
    posterior_estimator: Option<NeuralPosteriorEstimator>,
    classes: Vec<i32>,
    class_priors: HashMap<i32, f64>,
    config: DeepGenerativeConfig,
    fitted: bool,
}

#[derive(Debug, Clone)]
pub struct DeepGenerativeConfig {
    pub vae_config: VAEConfig,
    pub flow_config: FlowConfig,
    pub npe_config: NPEConfig,
    pub use_vae: bool,
    pub use_flow: bool,
    pub use_npe: bool,
}

impl Default for DeepGenerativeConfig {
    fn default() -> Self {
        Self {
            vae_config: VAEConfig::default(),
            flow_config: FlowConfig::default(),
            npe_config: NPEConfig::default(),
            use_vae: true,
            use_flow: false,
            use_npe: false,
        }
    }
}

impl DeepGenerativeNaiveBayes {
    pub fn new(config: DeepGenerativeConfig) -> Self {
        Self {
            vaes: HashMap::new(),
            flows: HashMap::new(),
            posterior_estimator: None,
            classes: Vec::new(),
            class_priors: HashMap::new(),
            config,
            fitted: false,
        }
    }

    /// Fit the deep generative model
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<(), DeepGenerativeError> {
        if x.nrows() != y.len() {
            return Err(DeepGenerativeError::DimensionMismatch {
                expected: x.nrows(),
                actual: y.len(),
            });
        }

        let input_dim = x.ncols();

        // Find unique classes and compute priors
        let mut unique_classes: Vec<i32> = y.iter().cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        self.classes = unique_classes;

        let n_samples = y.len() as f64;
        for &class in &self.classes {
            let class_count = y.iter().filter(|&&label| label == class).count() as f64;
            self.class_priors.insert(class, class_count / n_samples);
        }

        // Train models for each class
        for &class in &self.classes {
            // Get class-specific data
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            let class_data = Array2::from_shape_fn((class_indices.len(), input_dim), |(i, j)| {
                x[[class_indices[i], j]]
            });

            // Train VAE if enabled
            if self.config.use_vae {
                let vae = VariationalAutoencoder::new(input_dim, self.config.vae_config.clone());
                // Training logic would go here
                self.vaes.insert(class, vae);
            }

            // Train Flow if enabled
            if self.config.use_flow {
                let flow = NormalizingFlow::new(input_dim, self.config.flow_config.clone());
                // Training logic would go here
                self.flows.insert(class, flow);
            }
        }

        // Train Neural Posterior Estimator if enabled
        if self.config.use_npe {
            let npe =
                NeuralPosteriorEstimator::new(input_dim, input_dim, self.config.npe_config.clone());
            self.posterior_estimator = Some(npe);
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>, DeepGenerativeError> {
        if !self.fitted {
            return Err(DeepGenerativeError::PosteriorEstimationFailed(
                "Model not fitted".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut predictions = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let sample = x.row(i).to_owned();
            let mut class_probs = Array1::zeros(n_classes);

            for (j, &class) in self.classes.iter().enumerate() {
                let mut log_likelihood = 0.0;

                // Use VAE likelihood if available
                if let Some(vae) = self.vaes.get(&class) {
                    let (mean, logvar) = vae.encode(&sample);
                    let reconstruction = vae.decode(&mean);
                    log_likelihood += vae.compute_loss(&sample, &reconstruction, &mean, &logvar);
                }

                // Use Flow likelihood if available
                if let Some(flow) = self.flows.get(&class) {
                    log_likelihood += flow.log_prob(&sample);
                }

                // Add class prior
                if let Some(&prior) = self.class_priors.get(&class) {
                    log_likelihood += prior.ln();
                }

                class_probs[j] = log_likelihood;
            }

            // Normalize probabilities
            let max_log_prob = class_probs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            class_probs = class_probs.mapv(|x| (x - max_log_prob).exp());
            let total: f64 = class_probs.sum();
            if total > 0.0 {
                class_probs /= total;
            }

            predictions.row_mut(i).assign(&class_probs);
        }

        Ok(predictions)
    }

    /// Generate synthetic samples for a class
    pub fn generate_samples(
        &mut self,
        class: i32,
        n_samples: usize,
    ) -> Result<Array2<f64>, DeepGenerativeError> {
        if !self.fitted {
            return Err(DeepGenerativeError::PosteriorEstimationFailed(
                "Model not fitted".to_string(),
            ));
        }

        if let Some(vae) = self.vaes.get_mut(&class) {
            Ok(vae.generate(n_samples))
        } else if let Some(flow) = self.flows.get_mut(&class) {
            Ok(flow.sample(n_samples))
        } else {
            Err(DeepGenerativeError::PosteriorEstimationFailed(format!(
                "No generative model available for class {}",
                class
            )))
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_vae_creation() {
        let config = VAEConfig::default();
        let vae = VariationalAutoencoder::new(10, config);
        assert_eq!(vae.input_dim, 10);
        assert_eq!(vae.latent_dim, 10);
    }

    #[test]
    fn test_vae_encode_decode() {
        let config = VAEConfig::default();
        let mut vae = VariationalAutoencoder::new(5, config);

        let input = Array1::from_vec(vec![0.5, 0.3, 0.8, 0.1, 0.9]);
        let (mean, logvar) = vae.encode(&input);
        let z = vae.reparameterize(&mean, &logvar);
        let reconstruction = vae.decode(&z);

        assert_eq!(reconstruction.len(), 5);
        assert!(reconstruction.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_normalizing_flow_creation() {
        let config = FlowConfig::default();
        let flow = NormalizingFlow::new(5, config);
        assert_eq!(flow.coupling_layers.len(), 8);
    }

    #[test]
    fn test_flow_invertibility() {
        let config = FlowConfig::default();
        let flow = NormalizingFlow::new(4, config);

        let x = Array1::from_vec(vec![1.0, -0.5, 2.0, 0.3]);
        let (z, log_det_forward) = flow.forward(&x);
        let (x_reconstructed, log_det_inverse) = flow.inverse(&z);

        // Check invertibility
        for (orig, recon) in x.iter().zip(x_reconstructed.iter()) {
            assert_abs_diff_eq!(*orig, *recon, epsilon = 1e-6);
        }

        // Check log determinant consistency
        assert_abs_diff_eq!(log_det_forward, -log_det_inverse, epsilon = 1e-6);
    }

    #[test]
    fn test_neural_posterior_estimator() {
        let config = NPEConfig::default();
        let npe = NeuralPosteriorEstimator::new(3, 2, config);

        let params = Array1::from_vec(vec![0.5, -0.3, 1.2]);
        let obs = Array1::from_vec(vec![0.8, 0.1]);

        let log_prob = npe.log_posterior(&params, &obs);
        assert!(log_prob.is_finite());
    }

    #[test]
    fn test_deep_generative_nb_creation() {
        let config = DeepGenerativeConfig::default();
        let model = DeepGenerativeNaiveBayes::new(config);
        assert!(!model.fitted);
        assert_eq!(model.classes.len(), 0);
    }

    #[test]
    fn test_deep_generative_nb_fit() {
        let mut config = DeepGenerativeConfig::default();
        config.use_flow = false;
        config.use_npe = false;

        let mut model = DeepGenerativeNaiveBayes::new(config);

        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
                8.5, 9.5,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let result = model.fit(&x, &y);
        assert!(result.is_ok());
        assert!(model.fitted);
        assert_eq!(model.classes.len(), 2);
    }
}
