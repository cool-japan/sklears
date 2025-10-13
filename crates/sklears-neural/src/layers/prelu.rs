//! Parametric ReLU (PReLU) layer implementation.
//!
//! PReLU is a generalization of Leaky ReLU where the negative slope coefficient
//! is learned during training, making it more flexible and potentially more effective.

use crate::layers::{Layer, LayerConfig, ParameterizedLayer};
use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;
use std::marker::PhantomData;

/// Parametric ReLU layer with learnable negative slope parameters.
///
/// PReLU applies the function:
/// - f(x) = x if x > 0
/// - f(x) = α * x if x ≤ 0
///
/// where α is learned during training.
///
/// # Mathematical Formulation
///
/// Forward pass: `y_i = max(0, x_i) + α_i * min(0, x_i)`
/// Backward pass: `∂L/∂x_i = 1 if x_i > 0, else α_i`
/// Parameter update: `∂L/∂α_i = Σ(min(0, x_i) * ∂L/∂y_i)`
#[derive(Debug, Clone)]
pub struct PReLU<T: FloatBounds> {
    /// Learnable parameters (negative slope coefficients)
    alpha: Array1<T>,
    /// Gradients with respect to alpha parameters
    alpha_grad: Array1<T>,
    /// Last input values (stored for backward pass)
    last_input: Option<Array2<T>>,
    /// Number of channels/features
    num_parameters: usize,
    /// Configuration
    config: LayerConfig<T>,
    _phantom: PhantomData<T>,
}

impl<T: FloatBounds> PReLU<T> {
    /// Create a new PReLU layer.
    ///
    /// # Arguments
    /// * `num_parameters` - Number of parameters (channels). Use 1 for shared parameter across all channels.
    /// * `init_value` - Initial value for the alpha parameters (default: 0.25)
    pub fn new(num_parameters: usize, init_value: Option<T>) -> NeuralResult<Self> {
        let init_val =
            init_value.unwrap_or_else(|| T::from(0.25).unwrap_or(T::one() / T::from(4).unwrap()));

        Ok(Self {
            alpha: Array1::from_elem(num_parameters, init_val),
            alpha_grad: Array1::zeros(num_parameters),
            last_input: None,
            num_parameters,
            config: LayerConfig::new(num_parameters),
            _phantom: PhantomData,
        })
    }

    /// Create a PReLU layer with configuration.
    pub fn with_config(config: LayerConfig<T>, init_value: Option<T>) -> NeuralResult<Self> {
        Self::new(config.input_size, init_value)
    }

    /// Get the current alpha parameters.
    pub fn get_alpha(&self) -> &Array1<T> {
        &self.alpha
    }

    /// Set the alpha parameters.
    pub fn set_alpha(&mut self, alpha: Array1<T>) -> NeuralResult<()> {
        if alpha.len() != self.num_parameters {
            return Err(SklearsError::InvalidInput(format!(
                "Alpha array length {} doesn't match expected {}",
                alpha.len(),
                self.num_parameters
            )));
        }
        self.alpha = alpha;
        Ok(())
    }
}

impl<T: FloatBounds> Layer<T> for PReLU<T> {
    fn forward(&mut self, input: &Array2<T>, _training: bool) -> NeuralResult<Array2<T>> {
        let (batch_size, features) = input.dim();

        // Validate input dimensions
        if self.num_parameters != 1 && self.num_parameters != features {
            return Err(SklearsError::InvalidInput(format!(
                "PReLU expects {} parameters but got {} input features",
                self.num_parameters, features
            )));
        }

        // Store input for backward pass
        self.last_input = Some(input.clone());

        let mut output = Array2::zeros((batch_size, features));

        // Apply PReLU: f(x) = max(0, x) + alpha * min(0, x)
        for i in 0..batch_size {
            for j in 0..features {
                let x = input[[i, j]];
                let alpha_idx = if self.num_parameters == 1 { 0 } else { j };
                let alpha = self.alpha[alpha_idx];

                output[[i, j]] = if x > T::zero() { x } else { alpha * x };
            }
        }

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Array2<T>) -> NeuralResult<Array2<T>> {
        let input = self.last_input.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("No input stored for backward pass".to_string())
        })?;

        let (batch_size, features) = input.dim();
        let mut grad_input = Array2::zeros((batch_size, features));

        // Reset alpha gradients
        self.alpha_grad.fill(T::zero());

        // Compute gradients
        for i in 0..batch_size {
            for j in 0..features {
                let x = input[[i, j]];
                let grad_out = grad_output[[i, j]];
                let alpha_idx = if self.num_parameters == 1 { 0 } else { j };
                let alpha = self.alpha[alpha_idx];

                if x > T::zero() {
                    // Positive input: gradient flows through unchanged
                    grad_input[[i, j]] = grad_out;
                } else {
                    // Negative input: gradient scaled by alpha
                    grad_input[[i, j]] = alpha * grad_out;
                    // Accumulate alpha gradient: ∂L/∂α = x * ∂L/∂y
                    self.alpha_grad[alpha_idx] = self.alpha_grad[alpha_idx] + x * grad_out;
                }
            }
        }

        Ok(grad_input)
    }

    fn num_parameters(&self) -> usize {
        self.num_parameters
    }

    fn reset(&mut self) {
        self.last_input = None;
        self.alpha_grad.fill(T::zero());
    }
}

impl<T: FloatBounds> ParameterizedLayer<T> for PReLU<T> {
    fn parameters(&self) -> Vec<&Array1<T>> {
        vec![&self.alpha]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array1<T>> {
        vec![&mut self.alpha]
    }

    fn parameter_gradients(&self) -> Vec<Array1<T>> {
        vec![self.alpha_grad.clone()]
    }

    fn update_parameters(&mut self, updates: &[Array1<T>]) -> NeuralResult<()> {
        if updates.len() != 1 {
            return Err(SklearsError::InvalidInput(format!(
                "PReLU expects 1 parameter update, got {}",
                updates.len()
            )));
        }

        if updates[0].len() != self.alpha.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Update array length {} doesn't match parameter length {}",
                updates[0].len(),
                self.alpha.len()
            )));
        }

        // Apply parameter updates (typically from optimizer)
        self.alpha = &self.alpha + &updates[0];
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_prelu_creation() {
        let prelu: PReLU<f64> = PReLU::new(1, Some(0.1)).unwrap();
        assert_eq!(prelu.num_parameters(), 1);
        assert_abs_diff_eq!(prelu.get_alpha()[0], 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_prelu_forward_shared_parameter() {
        let mut prelu: PReLU<f64> = PReLU::new(1, Some(0.2)).unwrap();
        let input = array![[-2.0, -1.0, 0.0, 1.0, 2.0]];

        let output = prelu.forward(&input, true).unwrap();

        // Check positive values are unchanged
        assert_abs_diff_eq!(output[[0, 3]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(output[[0, 4]], 2.0, epsilon = 1e-10);

        // Check negative values are scaled by alpha
        assert_abs_diff_eq!(output[[0, 0]], -0.4, epsilon = 1e-10); // -2.0 * 0.2
        assert_abs_diff_eq!(output[[0, 1]], -0.2, epsilon = 1e-10); // -1.0 * 0.2

        // Check zero
        assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_prelu_forward_per_channel() {
        let mut prelu: PReLU<f64> = PReLU::new(3, Some(0.1)).unwrap();
        prelu.set_alpha(array![0.1, 0.2, 0.3]).unwrap();

        let input = array![[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]];
        let output = prelu.forward(&input, true).unwrap();

        // Check negative values are scaled by respective alphas
        assert_abs_diff_eq!(output[[0, 0]], -0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(output[[0, 1]], -0.2, epsilon = 1e-10);
        assert_abs_diff_eq!(output[[0, 2]], -0.3, epsilon = 1e-10);

        // Check positive values are unchanged
        assert_abs_diff_eq!(output[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(output[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(output[[1, 2]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_prelu_backward() {
        let mut prelu: PReLU<f64> = PReLU::new(1, Some(0.3)).unwrap();
        let input = array![[-2.0, 1.0], [-1.0, 2.0]];

        // Forward pass
        let _output = prelu.forward(&input, true).unwrap();

        // Backward pass
        let grad_output = array![[1.0, 1.0], [1.0, 1.0]];
        let grad_input = prelu.backward(&grad_output).unwrap();

        // Check input gradients
        assert_abs_diff_eq!(grad_input[[0, 0]], 0.3, epsilon = 1e-10); // negative: alpha
        assert_abs_diff_eq!(grad_input[[0, 1]], 1.0, epsilon = 1e-10); // positive: 1
        assert_abs_diff_eq!(grad_input[[1, 0]], 0.3, epsilon = 1e-10); // negative: alpha
        assert_abs_diff_eq!(grad_input[[1, 1]], 1.0, epsilon = 1e-10); // positive: 1

        // Check alpha gradients
        let alpha_grads = prelu.parameter_gradients();
        let expected_alpha_grad = -2.0 * 1.0 + -1.0 * 1.0; // sum of (x * grad_out) for negative x
        assert_abs_diff_eq!(alpha_grads[0][0], expected_alpha_grad, epsilon = 1e-10);
    }

    #[test]
    fn test_prelu_parameter_updates() {
        let mut prelu: PReLU<f64> = PReLU::new(2, Some(0.1)).unwrap();
        let original_alpha = prelu.get_alpha().clone();

        let updates = vec![array![0.05, 0.03]];
        prelu.update_parameters(&updates).unwrap();

        let new_alpha = prelu.get_alpha();
        assert_abs_diff_eq!(new_alpha[0], original_alpha[0] + 0.05, epsilon = 1e-10);
        assert_abs_diff_eq!(new_alpha[1], original_alpha[1] + 0.03, epsilon = 1e-10);
    }

    #[test]
    fn test_prelu_dimension_validation() {
        let mut prelu: PReLU<f64> = PReLU::new(3, Some(0.1)).unwrap();
        let input = array![[1.0, 2.0]]; // 2 features, but PReLU expects 3

        let result = prelu.forward(&input, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_prelu_gradient_check() {
        let mut prelu: PReLU<f64> = PReLU::new(1, Some(0.25)).unwrap();
        let input = array![[-1.5, 0.0, 1.5]];

        // Numerical gradient check
        let epsilon = 1e-7;
        let grad_output = array![[1.0, 1.0, 1.0]];

        // Forward pass
        let _output = prelu.forward(&input, true).unwrap();
        let _grad_input = prelu.backward(&grad_output).unwrap();
        let analytical_grad = prelu.parameter_gradients()[0][0];

        // Numerical gradient
        let mut prelu_plus = prelu.clone();
        let mut prelu_minus = prelu.clone();

        prelu_plus.alpha[0] = prelu.alpha[0] + epsilon;
        prelu_minus.alpha[0] = prelu.alpha[0] - epsilon;

        let output_plus = prelu_plus.forward(&input, true).unwrap();
        let output_minus = prelu_minus.forward(&input, true).unwrap();

        let loss_plus = (&output_plus * &grad_output).sum();
        let loss_minus = (&output_minus * &grad_output).sum();
        let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);

        assert_abs_diff_eq!(analytical_grad, numerical_grad, epsilon = 1e-5);
    }
}
