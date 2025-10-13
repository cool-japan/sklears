//! Activation functions for neural networks
//!
//! This module provides various activation functions commonly used in neural networks,
//! including ReLU, Sigmoid, Tanh, Linear, and Softmax activations.

// Use SciRS2-Core for arrays (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::types::Float;

/// Activation functions for neural networks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFunction {
    /// ReLU activation: max(0, x)
    ReLU,
    /// Sigmoid activation: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tanh,
    /// Linear activation: x (identity function)
    Linear,
    /// Softmax activation for multi-class outputs
    Softmax,
}

impl ActivationFunction {
    /// Apply activation function element-wise
    pub fn apply(&self, x: &Array1<Float>) -> Array1<Float> {
        match self {
            ActivationFunction::ReLU => x.map(|&val| val.max(0.0)),
            ActivationFunction::Sigmoid => x.map(|&val| 1.0 / (1.0 + (-val).exp())),
            ActivationFunction::Tanh => x.map(|&val| val.tanh()),
            ActivationFunction::Linear => x.clone(),
            ActivationFunction::Softmax => {
                let max_val = x.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
                let shifted = x.map(|&val| val - max_val);
                let exp_vals = shifted.map(|&val| val.exp());
                let sum_exp = exp_vals.sum();
                exp_vals.map(|&val| val / sum_exp)
            }
        }
    }

    /// Apply activation function to 2D array row-wise
    pub fn apply_2d(&self, x: &Array2<Float>) -> Array2<Float> {
        match self {
            ActivationFunction::Softmax => {
                let mut result = Array2::<Float>::zeros(x.dim());
                for (i, row) in x.axis_iter(Axis(0)).enumerate() {
                    let activated = self.apply(&row.to_owned());
                    result.row_mut(i).assign(&activated);
                }
                result
            }
            _ => x.map(|&val| {
                let single_val = Array1::from(vec![val]);
                self.apply(&single_val)[0]
            }),
        }
    }

    /// Compute derivative of activation function
    pub fn derivative(&self, x: &Array1<Float>) -> Array1<Float> {
        match self {
            ActivationFunction::ReLU => x.map(|&val| if val > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Sigmoid => {
                let sigmoid_vals = self.apply(x);
                sigmoid_vals.map(|&val| val * (1.0 - val))
            }
            ActivationFunction::Tanh => {
                let tanh_vals = self.apply(x);
                tanh_vals.map(|&val| 1.0 - val * val)
            }
            ActivationFunction::Linear => Array1::ones(x.len()),
            ActivationFunction::Softmax => {
                // For softmax, derivative is more complex and typically computed differently in practice
                Array1::ones(x.len())
            }
        }
    }
}
