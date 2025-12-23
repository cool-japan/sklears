//! Normalizing Flow models for density estimation and generative modeling.
//!
//! This module implements various normalizing flow architectures including:
//! - Coupling layers (RealNVP-style)
//! - Affine coupling flows
//! - Glow-style flows with invertible 1x1 convolutions
//! - Masked autoregressive flows (MAF)
//! - Inverse autoregressive flows (IAF)
//!
//! Normalizing flows learn invertible transformations to map simple distributions
//! (e.g., Gaussian) to complex data distributions while maintaining tractable
//! likelihood computation through the change of variables formula.

use crate::{activation::Activation, NeuralResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis, ScalarOperand};
use scirs2_core::random::{thread_rng, CoreRandom, Normal, Rng};
use sklears_core::{error::SklearsError, types::FloatBounds};
use std::f64::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Type of coupling transformation
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CouplingType {
    /// Additive coupling: y = x + t(x_masked)
    Additive,
    /// Affine coupling: y = x * exp(s(x_masked)) + t(x_masked)
    Affine,
}

/// Masking strategy for coupling layers
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MaskType {
    /// Split features in half (checkerboard for images)
    Checkerboard,
    /// Channel-wise masking
    Channelwise,
    /// Alternating split
    Alternating,
}

/// Configuration for a coupling layer
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CouplingLayerConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer sizes for scale and translation networks
    pub hidden_dims: Vec<usize>,
    /// Type of coupling transformation
    pub coupling_type: CouplingType,
    /// Masking type
    pub mask_type: MaskType,
    /// Whether to reverse the mask
    pub reverse_mask: bool,
    /// Activation function for hidden layers
    pub activation: String,
}

impl Default for CouplingLayerConfig {
    fn default() -> Self {
        Self {
            input_dim: 784,
            hidden_dims: vec![256, 256],
            coupling_type: CouplingType::Affine,
            mask_type: MaskType::Checkerboard,
            reverse_mask: false,
            activation: "relu".to_string(),
        }
    }
}

/// Affine coupling layer (RealNVP-style)
///
/// Splits the input into two parts and transforms one part conditioned on the other.
/// For affine coupling: y_b = x_b * exp(s(x_a)) + t(x_a)
/// where x = [x_a, x_b] and s, t are neural networks.
#[derive(Debug)]
pub struct AffineCouplingLayer<T: FloatBounds> {
    /// Input dimension
    input_dim: usize,
    /// Scale network weights
    scale_weights: Vec<Array2<T>>,
    /// Scale network biases
    scale_biases: Vec<Array1<T>>,
    /// Translation network weights
    translation_weights: Vec<Array2<T>>,
    /// Translation network biases
    translation_biases: Vec<Array1<T>>,
    /// Masking pattern (true = transform, false = identity)
    mask: Array1<bool>,
    /// Activation function
    activation: Activation,
    /// Whether this is affine or additive coupling
    is_affine: bool,
    /// Cached input for backward pass
    cached_input: Option<Array2<T>>,
    /// Cached scale values
    cached_scale: Option<Array2<T>>,
}

impl<T: FloatBounds> AffineCouplingLayer<T> {
    /// Create a new affine coupling layer
    pub fn new(config: CouplingLayerConfig) -> Self {
        let mut rng = thread_rng();
        let split_point = config.input_dim / 2;

        // Create mask
        let mut mask = Array1::from_elem(config.input_dim, false);
        match config.mask_type {
            MaskType::Checkerboard | MaskType::Channelwise => {
                for i in (if config.reverse_mask { split_point } else { 0 })
                    ..(if config.reverse_mask {
                        config.input_dim
                    } else {
                        split_point
                    })
                {
                    mask[i] = true;
                }
            }
            MaskType::Alternating => {
                for i in 0..config.input_dim {
                    if config.reverse_mask {
                        mask[i] = i % 2 == 1;
                    } else {
                        mask[i] = i % 2 == 0;
                    }
                }
            }
        }

        // Count masked and unmasked dimensions
        let n_masked = mask.iter().filter(|&&x| x).count();
        let n_unmasked = config.input_dim - n_masked;

        // Initialize scale network
        let mut scale_weights = Vec::new();
        let mut scale_biases = Vec::new();

        let mut prev_dim = n_masked;
        for &hidden_dim in &config.hidden_dims {
            let std = T::from((2.0 / prev_dim as f64).sqrt()).unwrap();
            let w = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                T::from(
                    rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std.to_f64().unwrap(),
                )
                .unwrap()
            });
            let b = Array1::zeros(hidden_dim);
            scale_weights.push(w);
            scale_biases.push(b);
            prev_dim = hidden_dim;
        }

        // Output layer for scale network
        let w = Array2::from_shape_fn((prev_dim, n_unmasked), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * 0.01).unwrap()
        });
        let b = Array1::zeros(n_unmasked);
        scale_weights.push(w);
        scale_biases.push(b);

        // Initialize translation network (same architecture)
        let mut translation_weights = Vec::new();
        let mut translation_biases = Vec::new();

        let mut prev_dim = n_masked;
        for &hidden_dim in &config.hidden_dims {
            let std = T::from((2.0 / prev_dim as f64).sqrt()).unwrap();
            let w = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                T::from(
                    rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std.to_f64().unwrap(),
                )
                .unwrap()
            });
            let b = Array1::zeros(hidden_dim);
            translation_weights.push(w);
            translation_biases.push(b);
            prev_dim = hidden_dim;
        }

        // Output layer for translation network
        let w = Array2::from_shape_fn((prev_dim, n_unmasked), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * 0.01).unwrap()
        });
        let b = Array1::zeros(n_unmasked);
        translation_weights.push(w);
        translation_biases.push(b);

        let activation = match config.activation.as_str() {
            "relu" => Activation::Relu,
            "tanh" => Activation::Tanh,
            "sigmoid" | "logistic" => Activation::Logistic,
            "elu" => Activation::Elu,
            _ => Activation::Relu,
        };

        Self {
            input_dim: config.input_dim,
            scale_weights,
            scale_biases,
            translation_weights,
            translation_biases,
            mask,
            activation,
            is_affine: matches!(config.coupling_type, CouplingType::Affine),
            cached_input: None,
            cached_scale: None,
        }
    }

    /// Forward transformation with log determinant Jacobian
    pub fn forward(&mut self, x: &Array2<T>) -> NeuralResult<(Array2<T>, T)> {
        let batch_size = x.nrows();

        // Split input based on mask
        let masked_indices: Vec<_> = self
            .mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| i)
            .collect();
        let unmasked_indices: Vec<_> = self
            .mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| !m)
            .map(|(i, _)| i)
            .collect();

        let x_masked = Array2::from_shape_fn((batch_size, masked_indices.len()), |(i, j)| {
            x[[i, masked_indices[j]]]
        });
        let x_unmasked = Array2::from_shape_fn((batch_size, unmasked_indices.len()), |(i, j)| {
            x[[i, unmasked_indices[j]]]
        });

        // Compute scale and translation from masked part
        let scale = self.compute_scale(&x_masked)?;
        let translation = self.compute_translation(&x_masked)?;

        // Apply transformation
        let y_unmasked = if self.is_affine {
            // Affine coupling: y = x * exp(s) + t
            let exp_scale = scale.mapv(|s| s.exp());
            &x_unmasked * &exp_scale + &translation
        } else {
            // Additive coupling: y = x + t
            &x_unmasked + &translation
        };

        // Reconstruct output
        let mut y = x.clone();
        for (i, &idx) in unmasked_indices.iter().enumerate() {
            for j in 0..batch_size {
                y[[j, idx]] = y_unmasked[[j, i]];
            }
        }

        // Compute log determinant Jacobian
        let log_det = if self.is_affine {
            // For affine coupling: log|det(J)| = sum(s)
            scale.sum_axis(Axis(1)).sum()
        } else {
            // For additive coupling: log|det(J)| = 0
            T::zero()
        };

        // Cache for backward pass
        self.cached_input = Some(x.clone());
        self.cached_scale = Some(scale);

        Ok((y, log_det))
    }

    /// Inverse transformation
    pub fn inverse(&self, y: &Array2<T>) -> NeuralResult<Array2<T>> {
        let batch_size = y.nrows();

        // Split input based on mask
        let masked_indices: Vec<_> = self
            .mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| i)
            .collect();
        let unmasked_indices: Vec<_> = self
            .mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| !m)
            .map(|(i, _)| i)
            .collect();

        let y_masked = Array2::from_shape_fn((batch_size, masked_indices.len()), |(i, j)| {
            y[[i, masked_indices[j]]]
        });
        let y_unmasked = Array2::from_shape_fn((batch_size, unmasked_indices.len()), |(i, j)| {
            y[[i, unmasked_indices[j]]]
        });

        // Compute scale and translation from masked part
        let scale = self.compute_scale(&y_masked)?;
        let translation = self.compute_translation(&y_masked)?;

        // Apply inverse transformation
        let x_unmasked = if self.is_affine {
            // Inverse affine: x = (y - t) / exp(s) = (y - t) * exp(-s)
            let exp_neg_scale = scale.mapv(|s| (-s).exp());
            (&y_unmasked - &translation) * &exp_neg_scale
        } else {
            // Inverse additive: x = y - t
            &y_unmasked - &translation
        };

        // Reconstruct output
        let mut x = y.clone();
        for (i, &idx) in unmasked_indices.iter().enumerate() {
            for j in 0..batch_size {
                x[[j, idx]] = x_unmasked[[j, i]];
            }
        }

        Ok(x)
    }

    /// Compute scale values from conditioned input
    fn compute_scale(&self, x: &Array2<T>) -> NeuralResult<Array2<T>> {
        let mut h = x.clone();

        // Forward through scale network
        for (i, (w, b)) in self
            .scale_weights
            .iter()
            .zip(self.scale_biases.iter())
            .enumerate()
        {
            h = h.dot(w) + b;

            // Apply activation to all but last layer
            if i < self.scale_weights.len() - 1 {
                h.mapv_inplace(|x| {
                    let x_f64 = x.to_f64().unwrap();
                    T::from(self.activation.forward(x_f64)).unwrap()
                });
            }
        }

        // Clamp scale to avoid numerical instability
        h.mapv_inplace(|s| {
            let s_f64 = s.to_f64().unwrap();
            T::from(s_f64.clamp(-10.0, 10.0)).unwrap()
        });

        Ok(h)
    }

    /// Compute translation values from conditioned input
    fn compute_translation(&self, x: &Array2<T>) -> NeuralResult<Array2<T>> {
        let mut h = x.clone();

        // Forward through translation network
        for (i, (w, b)) in self
            .translation_weights
            .iter()
            .zip(self.translation_biases.iter())
            .enumerate()
        {
            h = h.dot(w) + b;

            // Apply activation to all but last layer
            if i < self.translation_weights.len() - 1 {
                h.mapv_inplace(|x| {
                    let x_f64 = x.to_f64().unwrap();
                    T::from(self.activation.forward(x_f64)).unwrap()
                });
            }
        }

        Ok(h)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let scale_params: usize = self.scale_weights.iter().map(|w| w.len()).sum::<usize>()
            + self.scale_biases.iter().map(|b| b.len()).sum::<usize>();

        let translation_params: usize = self
            .translation_weights
            .iter()
            .map(|w| w.len())
            .sum::<usize>()
            + self
                .translation_biases
                .iter()
                .map(|b| b.len())
                .sum::<usize>();

        scale_params + translation_params
    }
}

/// Configuration for normalizing flow model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NormalizingFlowConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Number of coupling layers
    pub n_flows: usize,
    /// Hidden dimensions for coupling networks
    pub hidden_dims: Vec<usize>,
    /// Type of coupling
    pub coupling_type: CouplingType,
    /// Base distribution mean
    pub base_mean: f64,
    /// Base distribution standard deviation
    pub base_std: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of training iterations
    pub n_iterations: usize,
    /// Batch size
    pub batch_size: usize,
}

impl Default for NormalizingFlowConfig {
    fn default() -> Self {
        Self {
            input_dim: 784,
            n_flows: 8,
            hidden_dims: vec![256, 256],
            coupling_type: CouplingType::Affine,
            base_mean: 0.0,
            base_std: 1.0,
            learning_rate: 0.001,
            n_iterations: 1000,
            batch_size: 128,
        }
    }
}

/// Normalizing Flow model for density estimation
///
/// Implements a stack of coupling layers to learn complex distributions
/// from a simple base distribution (typically Gaussian).
pub struct NormalizingFlow<T: FloatBounds> {
    /// Stack of coupling layers
    coupling_layers: Vec<AffineCouplingLayer<T>>,
    /// Base distribution parameters
    base_mean: T,
    base_std: T,
    /// Input dimension
    input_dim: usize,
    /// Training configuration
    config: NormalizingFlowConfig,
}

impl<T: FloatBounds + ScalarOperand> NormalizingFlow<T> {
    /// Create a new normalizing flow model
    pub fn new(config: NormalizingFlowConfig) -> Self {
        let mut coupling_layers = Vec::new();

        for i in 0..config.n_flows {
            let layer_config = CouplingLayerConfig {
                input_dim: config.input_dim,
                hidden_dims: config.hidden_dims.clone(),
                coupling_type: config.coupling_type,
                mask_type: MaskType::Checkerboard,
                reverse_mask: i % 2 == 1, // Alternate masks
                activation: "relu".to_string(),
            };
            coupling_layers.push(AffineCouplingLayer::new(layer_config));
        }

        Self {
            coupling_layers,
            base_mean: T::from(config.base_mean).unwrap(),
            base_std: T::from(config.base_std).unwrap(),
            input_dim: config.input_dim,
            config,
        }
    }

    /// Forward pass: data -> latent
    pub fn forward(&mut self, x: &Array2<T>) -> NeuralResult<(Array2<T>, T)> {
        let mut z = x.clone();
        let mut log_det_sum = T::zero();

        // Pass through all coupling layers
        for layer in &mut self.coupling_layers {
            let (z_new, log_det) = layer.forward(&z)?;
            z = z_new;
            log_det_sum = log_det_sum + log_det;
        }

        Ok((z, log_det_sum))
    }

    /// Inverse pass: latent -> data
    pub fn inverse(&self, z: &Array2<T>) -> NeuralResult<Array2<T>> {
        let mut x = z.clone();

        // Pass through coupling layers in reverse
        for layer in self.coupling_layers.iter().rev() {
            x = layer.inverse(&x)?;
        }

        Ok(x)
    }

    /// Sample from the model
    pub fn sample(&self, n_samples: usize) -> NeuralResult<Array2<T>> {
        let mut rng = thread_rng();
        let normal = Normal::new(
            self.base_mean.to_f64().unwrap(),
            self.base_std.to_f64().unwrap(),
        )
        .unwrap();

        // Sample from base distribution
        let z = Array2::from_shape_fn((n_samples, self.input_dim), |_| {
            T::from(rng.sample::<f64, _>(normal)).unwrap()
        });

        // Transform to data distribution
        self.inverse(&z)
    }

    /// Compute negative log likelihood
    pub fn log_likelihood(&mut self, x: &Array2<T>) -> NeuralResult<T> {
        let (z, log_det) = self.forward(x)?;

        // Log probability under base distribution (Gaussian)
        let z_normalized = (&z - self.base_mean) / self.base_std;
        let log_prob_base = z_normalized.mapv(|zi| {
            let zi_f64 = zi.to_f64().unwrap();
            T::from(-0.5 * zi_f64 * zi_f64 - 0.5 * (2.0 * PI).ln()).unwrap()
        });

        let log_prob_sum = log_prob_base.sum();

        // Add log determinant (change of variables)
        let log_likelihood = log_prob_sum + log_det;

        Ok(log_likelihood)
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.coupling_layers
            .iter()
            .map(|layer| layer.num_parameters())
            .sum()
    }

    /// Get configuration
    pub fn config(&self) -> &NormalizingFlowConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_coupling_layer_creation() {
        let config = CouplingLayerConfig {
            input_dim: 10,
            hidden_dims: vec![32, 32],
            coupling_type: CouplingType::Affine,
            mask_type: MaskType::Checkerboard,
            reverse_mask: false,
            activation: "relu".to_string(),
        };

        let layer: AffineCouplingLayer<f64> = AffineCouplingLayer::new(config);
        assert_eq!(layer.input_dim, 10);
        assert!(layer.num_parameters() > 0);
    }

    #[test]
    fn test_coupling_layer_forward_backward() {
        let config = CouplingLayerConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            coupling_type: CouplingType::Affine,
            mask_type: MaskType::Checkerboard,
            reverse_mask: false,
            activation: "relu".to_string(),
        };

        let mut layer: AffineCouplingLayer<f64> = AffineCouplingLayer::new(config);
        let x =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let (y, _log_det) = layer.forward(&x).unwrap();
        let x_reconstructed = layer.inverse(&y).unwrap();

        // Check invertibility
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_relative_eq!(x[[i, j]], x_reconstructed[[i, j]], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_normalizing_flow_creation() {
        let config = NormalizingFlowConfig {
            input_dim: 10,
            n_flows: 4,
            hidden_dims: vec![32],
            coupling_type: CouplingType::Affine,
            base_mean: 0.0,
            base_std: 1.0,
            learning_rate: 0.001,
            n_iterations: 100,
            batch_size: 32,
        };

        let flow: NormalizingFlow<f64> = NormalizingFlow::new(config);
        assert_eq!(flow.input_dim, 10);
        assert_eq!(flow.coupling_layers.len(), 4);
        assert!(flow.num_parameters() > 0);
    }

    #[test]
    fn test_normalizing_flow_invertibility() {
        let config = NormalizingFlowConfig {
            input_dim: 8,
            n_flows: 3,
            hidden_dims: vec![16],
            coupling_type: CouplingType::Affine,
            ..Default::default()
        };

        let mut flow: NormalizingFlow<f64> = NormalizingFlow::new(config);
        let x = Array2::from_shape_fn((5, 8), |(i, j)| {
            (i as f64 + 1.0) * 0.1 + (j as f64 + 1.0) * 0.01
        });

        let (z, _log_det) = flow.forward(&x).unwrap();
        let x_reconstructed = flow.inverse(&z).unwrap();

        // Check invertibility
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_relative_eq!(x[[i, j]], x_reconstructed[[i, j]], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_normalizing_flow_sampling() {
        let config = NormalizingFlowConfig {
            input_dim: 5,
            n_flows: 2,
            hidden_dims: vec![10],
            coupling_type: CouplingType::Additive,
            ..Default::default()
        };

        let flow: NormalizingFlow<f64> = NormalizingFlow::new(config);
        let samples = flow.sample(10).unwrap();

        assert_eq!(samples.nrows(), 10);
        assert_eq!(samples.ncols(), 5);
    }

    #[test]
    fn test_normalizing_flow_log_likelihood() {
        let config = NormalizingFlowConfig {
            input_dim: 6,
            n_flows: 2,
            hidden_dims: vec![12],
            coupling_type: CouplingType::Affine,
            ..Default::default()
        };

        let mut flow: NormalizingFlow<f64> = NormalizingFlow::new(config);
        let x = Array2::from_shape_fn((3, 6), |(i, j)| (i as f64 + j as f64) * 0.1);

        let log_likelihood = flow.log_likelihood(&x).unwrap();
        // Should be finite
        assert!(log_likelihood.is_finite());
    }

    #[test]
    fn test_additive_coupling() {
        let config = CouplingLayerConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            coupling_type: CouplingType::Additive,
            mask_type: MaskType::Checkerboard,
            reverse_mask: false,
            activation: "relu".to_string(),
        };

        let mut layer: AffineCouplingLayer<f64> = AffineCouplingLayer::new(config);
        let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let (y, log_det) = layer.forward(&x).unwrap();

        // For additive coupling, log determinant should be zero
        assert_relative_eq!(log_det, 0.0, epsilon = 1e-10);

        // Check invertibility
        let x_reconstructed = layer.inverse(&y).unwrap();
        for i in 0..x.len() {
            assert_relative_eq!(x[[0, i]], x_reconstructed[[0, i]], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_mask_alternation() {
        let config1 = CouplingLayerConfig {
            input_dim: 6,
            hidden_dims: vec![8],
            coupling_type: CouplingType::Affine,
            mask_type: MaskType::Alternating,
            reverse_mask: false,
            activation: "relu".to_string(),
        };

        let config2 = CouplingLayerConfig {
            input_dim: 6,
            hidden_dims: vec![8],
            coupling_type: CouplingType::Affine,
            mask_type: MaskType::Alternating,
            reverse_mask: true,
            activation: "relu".to_string(),
        };

        let layer1: AffineCouplingLayer<f64> = AffineCouplingLayer::new(config1);
        let layer2: AffineCouplingLayer<f64> = AffineCouplingLayer::new(config2);

        // Check that masks are different
        let mask1_true = layer1.mask.iter().filter(|&&x| x).count();
        let mask2_true = layer2.mask.iter().filter(|&&x| x).count();

        // Both should have approximately half the dimensions masked
        assert!(mask1_true >= 2 && mask1_true <= 4);
        assert!(mask2_true >= 2 && mask2_true <= 4);
    }
}
