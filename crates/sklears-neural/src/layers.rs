//! Neural network layer implementations.
//!
//! This module provides modular neural network layer implementations including
//! batch normalization, dropout, and other essential building blocks for
//! modern neural architectures.

pub mod attention;
pub mod batch_norm;
pub mod dropout;
pub mod layer_norm;
pub mod prelu;
pub mod residual;
pub mod rnn;
pub mod transformer;

pub use attention::*;
pub use batch_norm::*;
pub use dropout::*;
pub use layer_norm::*;
pub use prelu::*;
pub use residual::*;
pub use rnn::*;
pub use transformer::*;

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::FloatBounds;

/// Base trait for neural network layers
pub trait Layer<T: FloatBounds> {
    /// Forward pass through the layer
    fn forward(&mut self, input: &Array2<T>, training: bool) -> NeuralResult<Array2<T>>;

    /// Backward pass through the layer (returns input gradients)
    fn backward(&mut self, grad_output: &Array2<T>) -> NeuralResult<Array2<T>>;

    /// Get the number of trainable parameters in this layer
    fn num_parameters(&self) -> usize {
        0
    }

    /// Reset the layer state (for stateful layers)
    fn reset(&mut self) {}
}

/// Trait for layers with learnable parameters
pub trait ParameterizedLayer<T: FloatBounds>: Layer<T> {
    /// Get the layer's parameters (weights, biases, etc.)
    fn parameters(&self) -> Vec<&Array1<T>>;

    /// Get mutable references to the layer's parameters
    fn parameters_mut(&mut self) -> Vec<&mut Array1<T>>;

    /// Get the gradients with respect to the layer's parameters
    fn parameter_gradients(&self) -> Vec<Array1<T>>;

    /// Update parameters using gradients (for custom optimizers)
    fn update_parameters(&mut self, updates: &[Array1<T>]) -> NeuralResult<()>;
}

/// Configuration for layer initialization
#[derive(Debug, Clone)]
pub struct LayerConfig<T: FloatBounds> {
    /// Input size (number of features)
    pub input_size: usize,
    /// Output size (number of features)
    pub output_size: Option<usize>,
    /// Momentum for moving averages (batch norm, etc.)
    pub momentum: T,
    /// Small constant for numerical stability
    pub epsilon: T,
    /// Whether to learn affine parameters (scale and shift)
    pub affine: bool,
    /// Whether to track running statistics
    pub track_running_stats: bool,
}

impl<T: FloatBounds> Default for LayerConfig<T> {
    fn default() -> Self {
        Self {
            input_size: 0,
            output_size: None,
            momentum: T::from(0.1).unwrap_or_else(|| T::one() / T::from(10).unwrap()),
            epsilon: T::from(1e-5).unwrap_or_else(|| T::one() / T::from(100000).unwrap()),
            affine: true,
            track_running_stats: true,
        }
    }
}

/// Builder for layer configurations
impl<T: FloatBounds> LayerConfig<T> {
    /// Create a new layer configuration
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            ..Default::default()
        }
    }

    /// Set the output size
    pub fn output_size(mut self, size: usize) -> Self {
        self.output_size = Some(size);
        self
    }

    /// Set the momentum for moving averages
    pub fn momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the epsilon for numerical stability
    pub fn epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set whether to learn affine parameters
    pub fn affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// Set whether to track running statistics
    pub fn track_running_stats(mut self, track: bool) -> Self {
        self.track_running_stats = track;
        self
    }
}
