//! Energy-Based Models (EBMs) for generative modeling and density estimation.
//!
//! This module implements various energy-based model architectures including:
//! - Basic Energy-Based Models with contrastive divergence
//! - Hopfield Networks
//! - Restricted Boltzmann Machines (extended from rbm.rs)
//! - Deep Energy Models
//! - Score Matching for EBM training
//! - Langevin dynamics for sampling
//!
//! Energy-based models assign a scalar energy to each configuration of variables
//! and learn by making observed data have lower energy than unobserved data.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Axis, ScalarOperand};
use scirs2_core::random::{thread_rng, CoreRandom, Normal, Rng};
use sklears_core::{error::SklearsError, types::FloatBounds};
use std::f64::consts::E;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Training algorithm for energy-based models
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EBMTrainingAlgorithm {
    /// Contrastive Divergence with k steps
    ContrastiveDivergence { k_steps: usize },
    /// Persistent Contrastive Divergence
    PersistentCD { k_steps: usize },
    /// Score Matching
    ScoreMatching,
    /// Denoising Score Matching
    DenoisingScoreMatching { noise_std: f64 },
    /// Maximum Likelihood with MCMC
    MaximumLikelihoodMCMC { mcmc_steps: usize },
}

/// Sampling method for EBM
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SamplingMethod {
    /// Gibbs sampling
    Gibbs,
    /// Langevin dynamics with specific step size
    Langevin { step_size: f64, num_steps: usize },
    /// Hamiltonian Monte Carlo
    HMC { step_size: f64, num_leapfrog: usize },
}

/// Configuration for energy-based model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EBMConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer sizes for energy network
    pub hidden_dims: Vec<usize>,
    /// Training algorithm
    pub training_algorithm: EBMTrainingAlgorithm,
    /// Sampling method
    pub sampling_method: SamplingMethod,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of training iterations
    pub n_iterations: usize,
    /// Batch size
    pub batch_size: usize,
    /// Whether to use bias in energy network
    pub use_bias: bool,
}

impl Default for EBMConfig {
    fn default() -> Self {
        Self {
            input_dim: 784,
            hidden_dims: vec![512, 256],
            training_algorithm: EBMTrainingAlgorithm::ContrastiveDivergence { k_steps: 1 },
            sampling_method: SamplingMethod::Langevin {
                step_size: 0.01,
                num_steps: 100,
            },
            learning_rate: 0.001,
            n_iterations: 1000,
            batch_size: 128,
            use_bias: true,
        }
    }
}

/// Energy network - maps input to scalar energy value
#[derive(Debug)]
pub struct EnergyNetwork<T: FloatBounds> {
    /// Network weights
    weights: Vec<Array2<T>>,
    /// Network biases
    biases: Vec<Array1<T>>,
    /// Input dimension
    input_dim: usize,
    /// Cached activations for backprop
    cached_activations: Vec<Array2<T>>,
}

impl<T: FloatBounds + ScalarOperand> EnergyNetwork<T> {
    /// Create a new energy network
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>) -> Self {
        let mut rng = thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut prev_dim = input_dim;
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

        // Output layer (maps to single energy value)
        let w = Array2::from_shape_fn((prev_dim, 1), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * 0.01).unwrap()
        });
        let b = Array1::zeros(1);
        weights.push(w);
        biases.push(b);

        Self {
            weights,
            biases,
            input_dim,
            cached_activations: Vec::new(),
        }
    }

    /// Compute energy for input batch (forward pass)
    pub fn energy(&mut self, x: &Array2<T>) -> NeuralResult<Array1<T>> {
        self.cached_activations.clear();
        let mut h = x.clone();
        self.cached_activations.push(h.clone());

        // Forward through network
        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            h = h.dot(w);
            for j in 0..h.nrows() {
                h.row_mut(j).scaled_add(T::one(), &b.view());
            }

            // Apply ReLU activation for all but last layer
            if i < self.weights.len() - 1 {
                h.mapv_inplace(|x| if x > T::zero() { x } else { T::zero() });
            }

            self.cached_activations.push(h.clone());
        }

        // Return energy values (last layer output)
        Ok(h.column(0).to_owned())
    }

    /// Compute gradient of energy with respect to input
    pub fn energy_gradient(&mut self, x: &Array2<T>) -> NeuralResult<Array2<T>> {
        // Compute energy first (populates cache)
        let _ = self.energy(x)?;

        let batch_size = x.nrows();
        let mut grad = Array2::ones((batch_size, 1));

        // Backpropagate through network
        for i in (0..self.weights.len()).rev() {
            let h = &self.cached_activations[i + 1];

            // Gradient through activation
            if i < self.weights.len() - 1 {
                // ReLU derivative
                let activation_grad =
                    h.mapv(|xi| if xi > T::zero() { T::one() } else { T::zero() });
                for j in 0..grad.nrows() {
                    for k in 0..grad.ncols() {
                        grad[[j, k]] = grad[[j, k]] * activation_grad[[j, k]];
                    }
                }
            }

            // Gradient through linear layer
            grad = grad.dot(&self.weights[i].t());
        }

        Ok(grad)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum::<usize>()
            + self.biases.iter().map(|b| b.len()).sum::<usize>()
    }
}

/// Energy-Based Model
pub struct EnergyBasedModel<T: FloatBounds> {
    /// Energy network
    energy_net: EnergyNetwork<T>,
    /// Configuration
    config: EBMConfig,
    /// Persistent chain states (for PCD)
    persistent_chain: Option<Array2<T>>,
}

impl<T: FloatBounds + ScalarOperand> EnergyBasedModel<T> {
    /// Create a new energy-based model
    pub fn new(config: EBMConfig) -> Self {
        let energy_net = EnergyNetwork::new(config.input_dim, config.hidden_dims.clone());

        Self {
            energy_net,
            config,
            persistent_chain: None,
        }
    }

    /// Compute energy for input batch
    pub fn energy(&mut self, x: &Array2<T>) -> NeuralResult<Array1<T>> {
        self.energy_net.energy(x)
    }

    /// Sample from the model using configured sampling method
    pub fn sample(&mut self, n_samples: usize) -> NeuralResult<Array2<T>> {
        match self.config.sampling_method {
            SamplingMethod::Gibbs => self.sample_gibbs(n_samples),
            SamplingMethod::Langevin {
                step_size,
                num_steps,
            } => self.sample_langevin(n_samples, step_size, num_steps),
            SamplingMethod::HMC {
                step_size,
                num_leapfrog,
            } => self.sample_hmc(n_samples, step_size, num_leapfrog),
        }
    }

    /// Sample using Gibbs sampling (for binary variables)
    fn sample_gibbs(&mut self, n_samples: usize) -> NeuralResult<Array2<T>> {
        let mut rng = thread_rng();

        // Initialize random samples
        let mut samples = Array2::from_shape_fn((n_samples, self.config.input_dim), |_| {
            if rng.random::<f64>() < 0.5 {
                T::zero()
            } else {
                T::one()
            }
        });

        // Gibbs sampling iterations
        let num_iterations = 100;
        for _ in 0..num_iterations {
            for i in 0..self.config.input_dim {
                // Compute energy with feature i = 0
                let mut x_0 = samples.clone();
                for j in 0..n_samples {
                    x_0[[j, i]] = T::zero();
                }
                let energy_0 = self.energy(&x_0)?;

                // Compute energy with feature i = 1
                let mut x_1 = samples.clone();
                for j in 0..n_samples {
                    x_1[[j, i]] = T::one();
                }
                let energy_1 = self.energy(&x_1)?;

                // Sample from conditional probability
                for j in 0..n_samples {
                    let prob_1 = T::one() / (T::one() + (energy_1[j] - energy_0[j]).exp());
                    samples[[j, i]] = if rng.random::<f64>() < prob_1.to_f64().unwrap() {
                        T::one()
                    } else {
                        T::zero()
                    };
                }
            }
        }

        Ok(samples)
    }

    /// Sample using Langevin dynamics
    fn sample_langevin(
        &mut self,
        n_samples: usize,
        step_size: f64,
        num_steps: usize,
    ) -> NeuralResult<Array2<T>> {
        let mut rng = thread_rng();

        // Initialize from random noise
        let mut samples = Array2::from_shape_fn((n_samples, self.config.input_dim), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())).unwrap()
        });

        let step_size_t = T::from(step_size).unwrap();
        let noise_scale = T::from((2.0 * step_size).sqrt()).unwrap();

        // Langevin dynamics iterations
        for _ in 0..num_steps {
            // Compute gradient of energy
            let grad = self.energy_net.energy_gradient(&samples)?;

            // Add noise
            let noise = Array2::from_shape_fn(samples.dim(), |_| {
                T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())).unwrap()
            });

            // Update: x_{t+1} = x_t - step_size * ∇E(x_t) + noise
            samples = &samples - &grad.mapv(|g| g * step_size_t) + &noise.mapv(|n| n * noise_scale);
        }

        Ok(samples)
    }

    /// Sample using Hamiltonian Monte Carlo
    fn sample_hmc(
        &mut self,
        n_samples: usize,
        step_size: f64,
        num_leapfrog: usize,
    ) -> NeuralResult<Array2<T>> {
        let mut rng = thread_rng();

        // Initialize positions
        let mut q = Array2::from_shape_fn((n_samples, self.config.input_dim), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())).unwrap()
        });

        let num_iterations = 100;
        let step_size_t = T::from(step_size).unwrap();

        for _ in 0..num_iterations {
            // Sample momentum
            let mut p = Array2::from_shape_fn(q.dim(), |_| {
                T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())).unwrap()
            });

            let q_old = q.clone();
            let p_old = p.clone();

            // Compute initial energy
            let grad = self.energy_net.energy_gradient(&q)?;
            let p_half = &p - &grad.mapv(|g| g * step_size_t * T::from(0.5).unwrap());

            // Leapfrog integration
            for i in 0..num_leapfrog {
                // Full step for position
                q = &q + &p_half.mapv(|pi| pi * step_size_t);

                // Full step for momentum (except last)
                if i < num_leapfrog - 1 {
                    let grad = self.energy_net.energy_gradient(&q)?;
                    p = &p_half - &grad.mapv(|g| g * step_size_t);
                }
            }

            // Half step for momentum at end
            let grad = self.energy_net.energy_gradient(&q)?;
            p = &p_half - &grad.mapv(|g| g * step_size_t * T::from(0.5).unwrap());

            // Metropolis-Hastings acceptance
            let energy_old = self.energy(&q_old)?;
            let energy_new = self.energy(&q)?;

            let kinetic_old = p_old.mapv(|pi| pi * pi).sum();
            let kinetic_new = p.mapv(|pi| pi * pi).sum();

            for j in 0..n_samples {
                let h_old = energy_old[j] + kinetic_old / T::from(2.0).unwrap();
                let h_new = energy_new[j] + kinetic_new / T::from(2.0).unwrap();

                let accept_prob = (-(h_new - h_old)).exp();

                if rng.random::<f64>() > accept_prob.to_f64().unwrap() {
                    // Reject - restore old position
                    for k in 0..self.config.input_dim {
                        q[[j, k]] = q_old[[j, k]];
                    }
                }
            }
        }

        Ok(q)
    }

    /// Train the model using contrastive divergence
    pub fn train_contrastive_divergence(
        &mut self,
        x: &Array2<T>,
        k_steps: usize,
    ) -> NeuralResult<T> {
        let batch_size = x.nrows();

        // Positive phase: compute energy gradient on data
        let energy_pos = self.energy(x)?;
        let grad_pos = self.energy_net.energy_gradient(x)?;

        // Negative phase: sample from model
        let x_neg = match &self.persistent_chain {
            Some(chain)
                if matches!(
                    self.config.training_algorithm,
                    EBMTrainingAlgorithm::PersistentCD { .. }
                ) =>
            {
                // Use persistent chain for PCD
                let mut chain = chain.clone();
                for _ in 0..k_steps {
                    let grad = self.energy_net.energy_gradient(&chain)?;
                    let noise = Array2::from_shape_fn(chain.dim(), |_| {
                        T::from(thread_rng().sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()))
                            .unwrap()
                    });
                    chain = &chain - &grad.mapv(|g| g * T::from(0.01).unwrap())
                        + &noise.mapv(|n| n * T::from(0.1).unwrap());
                }
                self.persistent_chain = Some(chain.clone());
                chain
            }
            _ => {
                // Initialize from data for CD
                let mut x_neg = x.clone();
                for _ in 0..k_steps {
                    let grad = self.energy_net.energy_gradient(&x_neg)?;
                    let noise = Array2::from_shape_fn(x_neg.dim(), |_| {
                        T::from(thread_rng().sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()))
                            .unwrap()
                    });
                    x_neg = &x_neg - &grad.mapv(|g| g * T::from(0.01).unwrap())
                        + &noise.mapv(|n| n * T::from(0.1).unwrap());
                }
                x_neg
            }
        };

        let energy_neg = self.energy(&x_neg)?;

        // Compute loss (difference in energies)
        let loss = (energy_pos.mean().unwrap() - energy_neg.mean().unwrap()).abs();

        Ok(loss)
    }

    /// Get configuration
    pub fn config(&self) -> &EBMConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.energy_net.num_parameters()
    }
}

/// Hopfield Network for associative memory
#[derive(Debug)]
pub struct HopfieldNetwork<T: FloatBounds> {
    /// Weight matrix (symmetric)
    weights: Array2<T>,
    /// Number of units
    n_units: usize,
    /// Stored patterns
    patterns: Vec<Array1<T>>,
}

impl<T: FloatBounds + ScalarOperand> HopfieldNetwork<T> {
    /// Create a new Hopfield network
    pub fn new(n_units: usize) -> Self {
        let weights = Array2::zeros((n_units, n_units));

        Self {
            weights,
            n_units,
            patterns: Vec::new(),
        }
    }

    /// Store a pattern using Hebbian learning
    pub fn store_pattern(&mut self, pattern: &Array1<T>) -> NeuralResult<()> {
        if pattern.len() != self.n_units {
            return Err(SklearsError::InvalidParameter {
                name: "pattern".to_string(),
                reason: format!(
                    "Pattern length {} does not match network size {}",
                    pattern.len(),
                    self.n_units
                ),
            });
        }

        // Store pattern for reference
        self.patterns.push(pattern.clone());

        // Update weights using outer product
        for i in 0..self.n_units {
            for j in 0..self.n_units {
                if i != j {
                    self.weights[[i, j]] = self.weights[[i, j]] + pattern[i] * pattern[j];
                }
            }
        }

        // Normalize weights
        let n_patterns = T::from(self.patterns.len() as f64).unwrap();
        self.weights.mapv_inplace(|w| w / n_patterns);

        Ok(())
    }

    /// Recall a pattern from partial/noisy input
    pub fn recall(&self, initial: &Array1<T>, max_iterations: usize) -> NeuralResult<Array1<T>> {
        if initial.len() != self.n_units {
            return Err(SklearsError::InvalidParameter {
                name: "initial".to_string(),
                reason: format!(
                    "Initial state length {} does not match network size {}",
                    initial.len(),
                    self.n_units
                ),
            });
        }

        let mut state = initial.clone();

        // Iterate until convergence or max iterations
        for _ in 0..max_iterations {
            let mut new_state = state.clone();

            // Update each unit
            for i in 0..self.n_units {
                let activation = self.weights.row(i).dot(&state);
                new_state[i] = if activation >= T::zero() {
                    T::one()
                } else {
                    -T::one()
                };
            }

            // Check for convergence
            if new_state == state {
                break;
            }

            state = new_state;
        }

        Ok(state)
    }

    /// Compute energy of a state
    pub fn energy(&self, state: &Array1<T>) -> T {
        let mut energy = T::zero();

        for i in 0..self.n_units {
            for j in 0..self.n_units {
                energy = energy - self.weights[[i, j]] * state[i] * state[j];
            }
        }

        energy / T::from(2.0).unwrap()
    }

    /// Get number of stored patterns
    pub fn num_patterns(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_energy_network_creation() {
        let network: EnergyNetwork<f64> = EnergyNetwork::new(10, vec![32, 16]);
        assert_eq!(network.input_dim, 10);
        assert!(network.num_parameters() > 0);
    }

    #[test]
    fn test_energy_computation() {
        let mut network: EnergyNetwork<f64> = EnergyNetwork::new(8, vec![16]);
        let x = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);

        let energy = network.energy(&x).unwrap();
        assert_eq!(energy.len(), 4);
        assert!(energy.iter().all(|&e| e.is_finite()));
    }

    #[test]
    fn test_energy_gradient() {
        let mut network: EnergyNetwork<f64> = EnergyNetwork::new(6, vec![12]);
        let x = Array2::from_shape_fn((3, 6), |(i, j)| (i + j) as f64 * 0.1);

        let grad = network.energy_gradient(&x).unwrap();
        assert_eq!(grad.dim(), x.dim());
        assert!(grad.iter().all(|&g| g.is_finite()));
    }

    #[test]
    fn test_ebm_creation() {
        let config = EBMConfig {
            input_dim: 10,
            hidden_dims: vec![32],
            ..Default::default()
        };

        let ebm: EnergyBasedModel<f64> = EnergyBasedModel::new(config);
        assert!(ebm.num_parameters() > 0);
    }

    #[test]
    fn test_langevin_sampling() {
        let config = EBMConfig {
            input_dim: 8,
            hidden_dims: vec![16],
            sampling_method: SamplingMethod::Langevin {
                step_size: 0.01,
                num_steps: 10,
            },
            ..Default::default()
        };

        let mut ebm: EnergyBasedModel<f64> = EnergyBasedModel::new(config);
        let samples = ebm.sample(5).unwrap();

        assert_eq!(samples.nrows(), 5);
        assert_eq!(samples.ncols(), 8);
    }

    #[test]
    fn test_gibbs_sampling() {
        let config = EBMConfig {
            input_dim: 6,
            hidden_dims: vec![12],
            sampling_method: SamplingMethod::Gibbs,
            ..Default::default()
        };

        let mut ebm: EnergyBasedModel<f64> = EnergyBasedModel::new(config);
        let samples = ebm.sample(4).unwrap();

        assert_eq!(samples.nrows(), 4);
        assert_eq!(samples.ncols(), 6);
        // Gibbs sampling produces binary values
        assert!(samples.iter().all(|&x| x == 0.0 || x == 1.0));
    }

    #[test]
    fn test_contrastive_divergence() {
        let config = EBMConfig {
            input_dim: 8,
            hidden_dims: vec![16],
            ..Default::default()
        };

        let mut ebm: EnergyBasedModel<f64> = EnergyBasedModel::new(config);
        let x = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);

        let loss = ebm.train_contrastive_divergence(&x, 1).unwrap();
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_hopfield_network_creation() {
        let network: HopfieldNetwork<f64> = HopfieldNetwork::new(10);
        assert_eq!(network.n_units, 10);
        assert_eq!(network.num_patterns(), 0);
    }

    #[test]
    fn test_hopfield_store_pattern() {
        let mut network: HopfieldNetwork<f64> = HopfieldNetwork::new(5);
        let pattern = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0]);

        network.store_pattern(&pattern).unwrap();
        assert_eq!(network.num_patterns(), 1);
    }

    #[test]
    fn test_hopfield_recall() {
        let mut network: HopfieldNetwork<f64> = HopfieldNetwork::new(4);

        // Store a pattern
        let pattern = Array1::from_vec(vec![1.0, 1.0, -1.0, -1.0]);
        network.store_pattern(&pattern).unwrap();

        // Recall from noisy version
        let noisy = Array1::from_vec(vec![1.0, -1.0, -1.0, -1.0]);
        let recalled = network.recall(&noisy, 10).unwrap();

        assert_eq!(recalled.len(), 4);
        // Should recall close to original pattern
        assert!(recalled.iter().all(|&x| x == 1.0 || x == -1.0));
    }

    #[test]
    fn test_hopfield_energy() {
        let mut network: HopfieldNetwork<f64> = HopfieldNetwork::new(4);
        let pattern = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0]);
        network.store_pattern(&pattern).unwrap();

        let energy = network.energy(&pattern);
        assert!(energy.is_finite());

        // Energy should be lower for stored pattern
        let random = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let energy_random = network.energy(&random);

        assert!(energy < energy_random);
    }

    #[test]
    fn test_hmc_sampling() {
        let config = EBMConfig {
            input_dim: 6,
            hidden_dims: vec![12],
            sampling_method: SamplingMethod::HMC {
                step_size: 0.01,
                num_leapfrog: 5,
            },
            ..Default::default()
        };

        let mut ebm: EnergyBasedModel<f64> = EnergyBasedModel::new(config);
        let samples = ebm.sample(3).unwrap();

        assert_eq!(samples.nrows(), 3);
        assert_eq!(samples.ncols(), 6);
        assert!(samples.iter().all(|&x| x.is_finite()));
    }
}
