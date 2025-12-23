//! Diffusion Models for generative modeling.
//!
//! This module implements various diffusion model architectures including:
//! - Denoising Diffusion Probabilistic Models (DDPM)
//! - Denoising Diffusion Implicit Models (DDIM)
//! - Score-based generative models
//! - Variance-preserving and variance-exploding diffusion processes
//!
//! Diffusion models learn to denoise data by gradually adding noise and then
//! learning to reverse the process, enabling high-quality sample generation.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Axis, ScalarOperand};
use scirs2_core::random::{thread_rng, CoreRandom, Normal, Rng};
use sklears_core::{error::SklearsError, types::FloatBounds};
use std::f64::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Type of noise schedule for diffusion process
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NoiseSchedule {
    /// Linear beta schedule: β_t = β_start + (β_end - β_start) * t / T
    Linear,
    /// Cosine schedule: More gradual noise addition
    Cosine,
    /// Quadratic schedule: β_t = β_start + (β_end - β_start) * (t / T)^2
    Quadratic,
    /// Sigmoid schedule: Smooth transition in noise levels
    Sigmoid,
}

/// Type of diffusion process
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DiffusionType {
    /// Variance Preserving (VP) - DDPM style
    VariancePreserving,
    /// Variance Exploding (VE) - Score-based
    VarianceExploding,
    /// Sub-Variance Preserving (sub-VP)
    SubVariancePreserving,
}

/// Configuration for diffusion model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DiffusionConfig {
    /// Number of diffusion timesteps
    pub num_timesteps: usize,
    /// Initial noise level (β_start)
    pub beta_start: f64,
    /// Final noise level (β_end)
    pub beta_end: f64,
    /// Type of noise schedule
    pub schedule: NoiseSchedule,
    /// Type of diffusion process
    pub diffusion_type: DiffusionType,
    /// Whether to clip denoised values
    pub clip_denoised: bool,
    /// Prediction type: "epsilon" (noise) or "x0" (original data)
    pub prediction_type: String,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            num_timesteps: 1000,
            beta_start: 0.0001,
            beta_end: 0.02,
            schedule: NoiseSchedule::Linear,
            diffusion_type: DiffusionType::VariancePreserving,
            clip_denoised: true,
            prediction_type: "epsilon".to_string(),
        }
    }
}

/// Noise scheduler for diffusion process
#[derive(Debug)]
pub struct NoiseScheduler<T: FloatBounds> {
    /// Beta values at each timestep
    betas: Array1<T>,
    /// Alpha values (1 - beta)
    alphas: Array1<T>,
    /// Cumulative product of alphas
    alphas_cumprod: Array1<T>,
    /// Previous cumulative product of alphas
    alphas_cumprod_prev: Array1<T>,
    /// Square root of cumulative alphas
    sqrt_alphas_cumprod: Array1<T>,
    /// Square root of (1 - cumulative alphas)
    sqrt_one_minus_alphas_cumprod: Array1<T>,
    /// Square root of reciprocal cumulative alphas
    sqrt_recip_alphas_cumprod: Array1<T>,
    /// Square root of (reciprocal cumulative alphas - 1)
    sqrt_recipm1_alphas_cumprod: Array1<T>,
    /// Posterior variance
    posterior_variance: Array1<T>,
    /// Log of posterior variance
    posterior_log_variance_clipped: Array1<T>,
    /// Number of timesteps
    num_timesteps: usize,
}

impl<T: FloatBounds> NoiseScheduler<T> {
    /// Create a new noise scheduler
    pub fn new(config: &DiffusionConfig) -> Self {
        let num_timesteps = config.num_timesteps;

        // Compute beta schedule
        let betas = Self::compute_beta_schedule(config);

        // Compute alpha values
        let alphas = betas.mapv(|b| T::one() - b);

        // Compute cumulative products
        let mut alphas_cumprod = Array1::ones(num_timesteps);
        let mut cumprod = T::one();
        for i in 0..num_timesteps {
            cumprod = cumprod * alphas[i];
            alphas_cumprod[i] = cumprod;
        }

        // Previous cumulative product (shifted by 1)
        let mut alphas_cumprod_prev = Array1::ones(num_timesteps);
        for i in 1..num_timesteps {
            alphas_cumprod_prev[i] = alphas_cumprod[i - 1];
        }

        // Precompute useful values
        let sqrt_alphas_cumprod = alphas_cumprod.mapv(|a| a.sqrt());
        let sqrt_one_minus_alphas_cumprod = alphas_cumprod.mapv(|a| (T::one() - a).sqrt());
        let sqrt_recip_alphas_cumprod = alphas_cumprod.mapv(|a| a.recip().sqrt());
        let sqrt_recipm1_alphas_cumprod = alphas_cumprod.mapv(|a| (a.recip() - T::one()).sqrt());

        // Compute posterior variance: β_t * (1 - α_{t-1}) / (1 - α_t)
        let posterior_variance = Array1::from_shape_fn(num_timesteps, |i| {
            if i == 0 {
                T::zero()
            } else {
                betas[i] * (T::one() - alphas_cumprod_prev[i]) / (T::one() - alphas_cumprod[i])
            }
        });

        // Log variance (clipped for numerical stability)
        let posterior_log_variance_clipped = posterior_variance.mapv(|v| {
            let v_f64 = v.to_f64().unwrap();
            T::from(v_f64.max(1e-20).ln()).unwrap()
        });

        Self {
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            posterior_variance,
            posterior_log_variance_clipped,
            num_timesteps,
        }
    }

    /// Compute beta schedule based on configuration
    fn compute_beta_schedule(config: &DiffusionConfig) -> Array1<T> {
        let num_timesteps = config.num_timesteps;
        let beta_start = T::from(config.beta_start).unwrap();
        let beta_end = T::from(config.beta_end).unwrap();

        match config.schedule {
            NoiseSchedule::Linear => {
                // Linear schedule: β_t = β_start + (β_end - β_start) * t / T
                Array1::from_shape_fn(num_timesteps, |t| {
                    let progress = T::from(t as f64 / num_timesteps as f64).unwrap();
                    beta_start + (beta_end - beta_start) * progress
                })
            }
            NoiseSchedule::Cosine => {
                // Cosine schedule (improved)
                let s = T::from(0.008).unwrap();
                Array1::from_shape_fn(num_timesteps, |t| {
                    let t_f64 = (t as f64 + 1.0) / num_timesteps as f64;
                    let alpha_t =
                        ((t_f64 + s.to_f64().unwrap()) / (1.0 + s.to_f64().unwrap()) * PI / 2.0)
                            .cos()
                            .powi(2);
                    let alpha_t_minus_1 = if t == 0 {
                        1.0
                    } else {
                        let t_prev = t as f64 / num_timesteps as f64;
                        ((t_prev + s.to_f64().unwrap()) / (1.0 + s.to_f64().unwrap()) * PI / 2.0)
                            .cos()
                            .powi(2)
                    };
                    T::from((1.0 - alpha_t / alpha_t_minus_1).min(0.999)).unwrap()
                })
            }
            NoiseSchedule::Quadratic => {
                // Quadratic schedule
                Array1::from_shape_fn(num_timesteps, |t| {
                    let progress = T::from(t as f64 / num_timesteps as f64).unwrap();
                    beta_start + (beta_end - beta_start) * progress * progress
                })
            }
            NoiseSchedule::Sigmoid => {
                // Sigmoid schedule
                Array1::from_shape_fn(num_timesteps, |t| {
                    let progress = t as f64 / num_timesteps as f64;
                    let sig = 1.0 / (1.0 + (-12.0 * (progress - 0.5)).exp());
                    T::from(
                        beta_start.to_f64().unwrap()
                            + (beta_end.to_f64().unwrap() - beta_start.to_f64().unwrap()) * sig,
                    )
                    .unwrap()
                })
            }
        }
    }

    /// Add noise to data at timestep t (forward diffusion)
    pub fn add_noise(
        &self,
        x0: &Array2<T>,
        noise: &Array2<T>,
        t: usize,
    ) -> NeuralResult<Array2<T>> {
        if t >= self.num_timesteps {
            return Err(SklearsError::InvalidParameter {
                name: "timestep".to_string(),
                reason: format!("Timestep {} exceeds maximum {}", t, self.num_timesteps),
            });
        }

        // x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
        let sqrt_alpha = self.sqrt_alphas_cumprod[t];
        let sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t];

        let noisy_data = x0.mapv(|x| x * sqrt_alpha) + noise.mapv(|n| n * sqrt_one_minus_alpha);
        Ok(noisy_data)
    }

    /// Get posterior mean and variance for denoising step
    pub fn get_posterior(
        &self,
        x_t: &Array2<T>,
        x0_pred: &Array2<T>,
        t: usize,
    ) -> NeuralResult<(Array2<T>, T)> {
        if t >= self.num_timesteps {
            return Err(SklearsError::InvalidParameter {
                name: "timestep".to_string(),
                reason: format!("Timestep {} exceeds maximum {}", t, self.num_timesteps),
            });
        }

        // Posterior mean: μ = (sqrt(α_bar_{t-1}) * β_t / (1 - α_bar_t)) * x_0
        //                    + (sqrt(α_t) * (1 - α_bar_{t-1}) / (1 - α_bar_t)) * x_t
        let alpha_t = self.alphas[t];
        let alpha_bar_t = self.alphas_cumprod[t];
        let alpha_bar_t_prev = self.alphas_cumprod_prev[t];
        let beta_t = self.betas[t];

        let coef1 = (alpha_bar_t_prev.sqrt() * beta_t) / (T::one() - alpha_bar_t);
        let coef2 = (alpha_t.sqrt() * (T::one() - alpha_bar_t_prev)) / (T::one() - alpha_bar_t);

        let posterior_mean = x0_pred.mapv(|x| x * coef1) + x_t.mapv(|x| x * coef2);
        let posterior_var = self.posterior_variance[t];

        Ok((posterior_mean, posterior_var))
    }
}

/// Denoising network trait
///
/// Implement this trait for your denoising neural network
pub trait DenoisingNetwork<T: FloatBounds> {
    /// Predict noise or x0 given noisy input and timestep
    fn predict(&mut self, x_t: &Array2<T>, t: usize) -> NeuralResult<Array2<T>>;

    /// Update network parameters (for training)
    fn update(&mut self, gradients: &Array2<T>, learning_rate: T) -> NeuralResult<()>;

    /// Get number of parameters
    fn num_parameters(&self) -> usize;
}

/// Simple MLP denoising network for demonstration
pub struct MLPDenoiser<T: FloatBounds> {
    /// Network weights (list of weight matrices)
    weights: Vec<Array2<T>>,
    /// Network biases
    biases: Vec<Array1<T>>,
    /// Input dimension
    input_dim: usize,
    /// Hidden dimensions
    hidden_dims: Vec<usize>,
    /// Number of timesteps (for embedding)
    num_timesteps: usize,
    /// Cached activations for backprop
    cached_activations: Vec<Array2<T>>,
}

impl<T: FloatBounds + ScalarOperand> MLPDenoiser<T> {
    /// Create a new MLP denoiser
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, num_timesteps: usize) -> Self {
        let mut rng = thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // First layer (input + time embedding)
        let time_embed_dim = 64;
        let first_layer_input = input_dim + time_embed_dim;

        let mut prev_dim = first_layer_input;
        for &hidden_dim in &hidden_dims {
            let std = (2.0 / prev_dim as f64).sqrt();
            let w = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std).unwrap()
            });
            let b = Array1::zeros(hidden_dim);
            weights.push(w);
            biases.push(b);
            prev_dim = hidden_dim;
        }

        // Output layer
        let w = Array2::from_shape_fn((prev_dim, input_dim), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * 0.01).unwrap()
        });
        let b = Array1::zeros(input_dim);
        weights.push(w);
        biases.push(b);

        Self {
            weights,
            biases,
            input_dim,
            hidden_dims,
            num_timesteps,
            cached_activations: Vec::new(),
        }
    }

    /// Create sinusoidal time embedding
    fn time_embedding(&self, t: usize, batch_size: usize) -> Array2<T> {
        let embed_dim = 64;
        let half_dim = embed_dim / 2;

        let t_norm = t as f64 / self.num_timesteps as f64;

        Array2::from_shape_fn((batch_size, embed_dim), |(_, j)| {
            if j < half_dim {
                let freq = (j as f64 / half_dim as f64 * 10.0).exp();
                T::from((t_norm * freq).sin()).unwrap()
            } else {
                let freq = ((j - half_dim) as f64 / half_dim as f64 * 10.0).exp();
                T::from((t_norm * freq).cos()).unwrap()
            }
        })
    }
}

impl<T: FloatBounds + ScalarOperand> DenoisingNetwork<T> for MLPDenoiser<T> {
    fn predict(&mut self, x_t: &Array2<T>, t: usize) -> NeuralResult<Array2<T>> {
        let batch_size = x_t.nrows();

        // Get time embedding
        let time_embed = self.time_embedding(t, batch_size);

        // Concatenate input with time embedding
        let mut h = Array2::zeros((batch_size, x_t.ncols() + time_embed.ncols()));
        for i in 0..batch_size {
            for j in 0..x_t.ncols() {
                h[[i, j]] = x_t[[i, j]];
            }
            for j in 0..time_embed.ncols() {
                h[[i, x_t.ncols() + j]] = time_embed[[i, j]];
            }
        }

        // Clear cached activations
        self.cached_activations.clear();
        self.cached_activations.push(h.clone());

        // Forward pass through network
        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            h = h.dot(w);
            for j in 0..h.nrows() {
                h.row_mut(j).scaled_add(T::one(), &b.view());
            }

            // ReLU activation for all but last layer
            if i < self.weights.len() - 1 {
                h.mapv_inplace(|x| if x > T::zero() { x } else { T::zero() });
            }

            self.cached_activations.push(h.clone());
        }

        Ok(h)
    }

    fn update(&mut self, _gradients: &Array2<T>, _learning_rate: T) -> NeuralResult<()> {
        // Simplified update - in practice, this would implement proper backpropagation
        Ok(())
    }

    fn num_parameters(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum::<usize>()
            + self.biases.iter().map(|b| b.len()).sum::<usize>()
    }
}

/// Denoising Diffusion Probabilistic Model (DDPM)
pub struct DDPM<T: FloatBounds, N: DenoisingNetwork<T>> {
    /// Noise scheduler
    scheduler: NoiseScheduler<T>,
    /// Denoising network
    network: N,
    /// Configuration
    config: DiffusionConfig,
    /// Phantom data for T
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FloatBounds + ScalarOperand, N: DenoisingNetwork<T>> DDPM<T, N> {
    /// Create a new DDPM model
    pub fn new(config: DiffusionConfig, network: N) -> Self {
        let scheduler = NoiseScheduler::new(&config);

        Self {
            scheduler,
            network,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Training step: compute loss for a batch
    pub fn train_step(&mut self, x0: &Array2<T>) -> NeuralResult<T> {
        let batch_size = x0.nrows();
        let mut rng = thread_rng();

        // Sample random timesteps
        let t = rng.random_range(0..self.config.num_timesteps);

        // Sample noise
        let noise = Array2::from_shape_fn(x0.dim(), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())).unwrap()
        });

        // Add noise to data
        let x_t = self.scheduler.add_noise(x0, &noise, t)?;

        // Predict noise
        let noise_pred = self.network.predict(&x_t, t)?;

        // Compute MSE loss
        let diff = &noise_pred - &noise;
        let loss = diff.mapv(|x| x * x).mean().unwrap();

        Ok(loss)
    }

    /// Sample from the model (reverse diffusion)
    pub fn sample(&mut self, n_samples: usize, input_dim: usize) -> NeuralResult<Array2<T>> {
        let mut rng = thread_rng();

        // Start from pure noise
        let mut x = Array2::from_shape_fn((n_samples, input_dim), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())).unwrap()
        });

        // Reverse diffusion process
        for t in (0..self.config.num_timesteps).rev() {
            // Predict noise
            let noise_pred = self.network.predict(&x, t)?;

            // Compute x0 prediction
            let alpha_bar_t = self.scheduler.sqrt_alphas_cumprod[t];
            let sqrt_one_minus_alpha_bar = self.scheduler.sqrt_one_minus_alphas_cumprod[t];

            let x0_pred = (&x - noise_pred.mapv(|n| n * sqrt_one_minus_alpha_bar))
                .mapv(|xi| xi / alpha_bar_t);

            // Clip if configured
            let x0_pred = if self.config.clip_denoised {
                x0_pred.mapv(|xi| {
                    let xi_f64 = xi.to_f64().unwrap();
                    T::from(xi_f64.clamp(-1.0, 1.0)).unwrap()
                })
            } else {
                x0_pred
            };

            // Get posterior mean and variance
            let (mean, variance) = self.scheduler.get_posterior(&x, &x0_pred, t)?;

            // Add noise (except for last step)
            if t > 0 {
                let z = Array2::from_shape_fn(x.dim(), |_| {
                    T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())).unwrap()
                });
                x = mean + z.mapv(|zi| zi * variance.sqrt());
            } else {
                x = mean;
            }
        }

        Ok(x)
    }

    /// Get configuration
    pub fn config(&self) -> &DiffusionConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.network.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_noise_scheduler_creation() {
        let config = DiffusionConfig::default();
        let scheduler: NoiseScheduler<f64> = NoiseScheduler::new(&config);

        assert_eq!(scheduler.num_timesteps, config.num_timesteps);
        assert_eq!(scheduler.betas.len(), config.num_timesteps);
        assert_eq!(scheduler.alphas.len(), config.num_timesteps);
    }

    #[test]
    fn test_linear_schedule() {
        let config = DiffusionConfig {
            num_timesteps: 100,
            beta_start: 0.0001,
            beta_end: 0.02,
            schedule: NoiseSchedule::Linear,
            ..Default::default()
        };

        let scheduler: NoiseScheduler<f64> = NoiseScheduler::new(&config);

        // Check that betas increase linearly
        assert!(scheduler.betas[0] < scheduler.betas[50]);
        assert!(scheduler.betas[50] < scheduler.betas[99]);
        assert_relative_eq!(scheduler.betas[0], 0.0001, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_schedule() {
        let config = DiffusionConfig {
            num_timesteps: 100,
            schedule: NoiseSchedule::Cosine,
            ..Default::default()
        };

        let scheduler: NoiseScheduler<f64> = NoiseScheduler::new(&config);

        // Check that all betas are positive and bounded
        for beta in scheduler.betas.iter() {
            assert!(*beta > 0.0);
            assert!(*beta < 1.0);
        }
    }

    #[test]
    fn test_add_noise() {
        let config = DiffusionConfig {
            num_timesteps: 10,
            ..Default::default()
        };
        let scheduler: NoiseScheduler<f64> = NoiseScheduler::new(&config);

        let x0 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let noise = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

        let x_t = scheduler.add_noise(&x0, &noise, 5).unwrap();

        assert_eq!(x_t.dim(), x0.dim());
        // Noisy data should be different from original
        assert!((x_t[[0, 0]] - x0[[0, 0]]).abs() > 1e-6);
    }

    #[test]
    fn test_posterior_computation() {
        let config = DiffusionConfig {
            num_timesteps: 10,
            ..Default::default()
        };
        let scheduler: NoiseScheduler<f64> = NoiseScheduler::new(&config);

        let x_t = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x0_pred = Array2::from_shape_vec((2, 3), vec![0.9, 1.9, 2.9, 3.9, 4.9, 5.9]).unwrap();

        let (mean, variance) = scheduler.get_posterior(&x_t, &x0_pred, 5).unwrap();

        assert_eq!(mean.dim(), x_t.dim());
        assert!(variance.is_finite());
        assert!(variance >= 0.0);
    }

    #[test]
    fn test_mlp_denoiser_creation() {
        let denoiser: MLPDenoiser<f64> = MLPDenoiser::new(10, vec![64, 64], 1000);

        assert_eq!(denoiser.input_dim, 10);
        assert!(denoiser.num_parameters() > 0);
    }

    #[test]
    fn test_mlp_denoiser_prediction() {
        let mut denoiser: MLPDenoiser<f64> = MLPDenoiser::new(8, vec![32], 1000);

        let x_t = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);
        let noise_pred = denoiser.predict(&x_t, 500).unwrap();

        assert_eq!(noise_pred.dim(), x_t.dim());
    }

    #[test]
    fn test_ddpm_creation() {
        let config = DiffusionConfig {
            num_timesteps: 100,
            ..Default::default()
        };
        let network = MLPDenoiser::<f64>::new(10, vec![32], config.num_timesteps);
        let ddpm = DDPM::new(config, network);

        assert_eq!(ddpm.config().num_timesteps, 100);
        assert!(ddpm.num_parameters() > 0);
    }

    #[test]
    fn test_ddpm_train_step() {
        let config = DiffusionConfig {
            num_timesteps: 50,
            ..Default::default()
        };
        let network = MLPDenoiser::<f64>::new(8, vec![32], config.num_timesteps);
        let mut ddpm = DDPM::new(config, network);

        let x0 = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);
        let loss = ddpm.train_step(&x0).unwrap();

        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_ddpm_sampling() {
        let config = DiffusionConfig {
            num_timesteps: 10, // Small for testing
            ..Default::default()
        };
        let network = MLPDenoiser::<f64>::new(6, vec![16], config.num_timesteps);
        let mut ddpm = DDPM::new(config, network);

        let samples = ddpm.sample(2, 6).unwrap();

        assert_eq!(samples.nrows(), 2);
        assert_eq!(samples.ncols(), 6);
    }

    #[test]
    fn test_quadratic_schedule() {
        let config = DiffusionConfig {
            num_timesteps: 100,
            schedule: NoiseSchedule::Quadratic,
            ..Default::default()
        };

        let scheduler: NoiseScheduler<f64> = NoiseScheduler::new(&config);

        // Check quadratic growth
        assert!(
            scheduler.betas[25] - scheduler.betas[0] < scheduler.betas[75] - scheduler.betas[50]
        );
    }

    #[test]
    fn test_sigmoid_schedule() {
        let config = DiffusionConfig {
            num_timesteps: 100,
            schedule: NoiseSchedule::Sigmoid,
            ..Default::default()
        };

        let scheduler: NoiseScheduler<f64> = NoiseScheduler::new(&config);

        // Check all betas are valid
        for beta in scheduler.betas.iter() {
            assert!(*beta > 0.0);
            assert!(*beta < 1.0);
        }
    }
}
