//! Residual connection implementations for deep neural networks.
//!
//! Residual connections, introduced in ResNet, allow gradients to flow directly
//! through shortcut connections, enabling training of much deeper networks.

use crate::layers::{Layer, LayerConfig};
use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;
use std::marker::PhantomData;

/// Types of residual connections
#[derive(Debug, Clone, PartialEq)]
pub enum ResidualType {
    /// Simple addition: output = input + layer_output
    Addition,
    /// Concatenation: output = [input, layer_output]
    Concatenation,
    /// Gated addition with learnable gate
    Gated,
}

/// Residual connection layer that adds skip connections to enable deep networks.
///
/// # Mathematical Formulation
///
/// For Addition type: `output = input + F(input)`
/// For Concatenation type: `output = [input; F(input)]`
/// For Gated type: `output = α * input + (1-α) * F(input)` where α is learnable
///
/// Where F(input) is the output of the wrapped layer.
#[derive(Debug, Clone)]
pub struct ResidualBlock<T: FloatBounds> {
    /// Type of residual connection
    residual_type: ResidualType,
    /// Whether the layer can handle dimension mismatches
    adaptive: bool,
    /// Projection layer for dimension matching (if needed)
    projection: Option<LinearProjection<T>>,
    /// Gate parameter for gated residual connections
    gate: Option<T>,
    /// Last input for backward pass
    last_input: Option<Array2<T>>,
    /// Last layer output for backward pass
    last_layer_output: Option<Array2<T>>,
    /// Configuration
    config: LayerConfig<T>,
    _phantom: PhantomData<T>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> ResidualBlock<T> {
    /// Create a new residual block.
    ///
    /// # Arguments
    /// * `residual_type` - Type of residual connection
    /// * `adaptive` - Whether to automatically handle dimension mismatches
    pub fn new(residual_type: ResidualType, adaptive: bool) -> Self {
        let gate = if matches!(residual_type, ResidualType::Gated) {
            Some(T::from(0.5).unwrap_or_else(|| T::one() / T::from(2).unwrap()))
        } else {
            None
        };

        Self {
            residual_type,
            adaptive,
            projection: None,
            gate,
            last_input: None,
            last_layer_output: None,
            config: LayerConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Create a residual block with configuration.
    pub fn with_config(config: LayerConfig<T>, residual_type: ResidualType) -> Self {
        let mut block = Self::new(residual_type, true);
        block.config = config;
        block
    }

    /// Set the gate parameter for gated residual connections.
    pub fn set_gate(&mut self, gate: T) -> NeuralResult<()> {
        if !matches!(self.residual_type, ResidualType::Gated) {
            return Err(SklearsError::InvalidInput(
                "Gate can only be set for gated residual connections".to_string(),
            ));
        }
        self.gate = Some(gate);
        Ok(())
    }

    /// Get the current gate parameter.
    pub fn get_gate(&self) -> Option<T> {
        self.gate
    }

    /// Process the residual connection given input and layer output.
    pub fn apply_residual(
        &mut self,
        input: &Array2<T>,
        layer_output: &Array2<T>,
    ) -> NeuralResult<Array2<T>> {
        // Store for backward pass
        self.last_input = Some(input.clone());
        self.last_layer_output = Some(layer_output.clone());

        let (input_shape, layer_shape) = (input.dim(), layer_output.dim());

        match self.residual_type {
            ResidualType::Addition => {
                if input_shape != layer_shape {
                    if self.adaptive {
                        // Create or use projection layer to match dimensions
                        if self.projection.is_none() {
                            self.projection =
                                Some(LinearProjection::new(input_shape.1, layer_shape.1)?);
                        }

                        let projected_input =
                            self.projection.as_mut().unwrap().forward(input, false)?;
                        Ok(&projected_input + layer_output)
                    } else {
                        Err(SklearsError::InvalidInput(format!(
                            "Dimension mismatch: input {:?} vs layer output {:?}",
                            input_shape, layer_shape
                        )))
                    }
                } else {
                    Ok(input + layer_output)
                }
            }
            ResidualType::Concatenation => {
                if input_shape.0 != layer_shape.0 {
                    return Err(SklearsError::InvalidInput(format!(
                        "Batch size mismatch: input {} vs layer output {}",
                        input_shape.0, layer_shape.0
                    )));
                }

                // Concatenate along feature dimension
                let mut result = Array2::zeros((input_shape.0, input_shape.1 + layer_shape.1));
                result.slice_mut(s![.., ..input_shape.1]).assign(input);
                result
                    .slice_mut(s![.., input_shape.1..])
                    .assign(layer_output);
                Ok(result)
            }
            ResidualType::Gated => {
                let gate = self.gate.unwrap_or_else(|| {
                    T::from(0.5).unwrap_or_else(|| T::one() / T::from(2).unwrap())
                });

                if input_shape != layer_shape {
                    if self.adaptive {
                        if self.projection.is_none() {
                            self.projection =
                                Some(LinearProjection::new(input_shape.1, layer_shape.1)?);
                        }

                        let projected_input =
                            self.projection.as_mut().unwrap().forward(input, false)?;
                        Ok(&projected_input * gate + layer_output * (T::one() - gate))
                    } else {
                        Err(SklearsError::InvalidInput(format!(
                            "Dimension mismatch: input {:?} vs layer output {:?}",
                            input_shape, layer_shape
                        )))
                    }
                } else {
                    Ok(input * gate + layer_output * (T::one() - gate))
                }
            }
        }
    }
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> Layer<T> for ResidualBlock<T> {
    fn forward(&mut self, input: &Array2<T>, _training: bool) -> NeuralResult<Array2<T>> {
        // For a standalone residual block, we just return the input
        // In practice, this would be used in combination with other layers
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Array2<T>) -> NeuralResult<Array2<T>> {
        let input = self.last_input.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("No input stored for backward pass".to_string())
        })?;

        let layer_output = self.last_layer_output.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("No layer output stored for backward pass".to_string())
        })?;

        match self.residual_type {
            ResidualType::Addition => {
                // Gradient flows to both input and layer output
                if input.dim() == layer_output.dim() {
                    Ok(grad_output.clone())
                } else if let Some(ref mut projection) = self.projection {
                    // Gradient flows through projection
                    projection.backward(grad_output)
                } else {
                    Ok(grad_output.clone())
                }
            }
            ResidualType::Concatenation => {
                // Split gradient between input and layer output
                let input_features = input.dim().1;
                Ok(grad_output.slice(s![.., ..input_features]).to_owned())
            }
            ResidualType::Gated => {
                let gate = self.gate.unwrap_or_else(|| {
                    T::from(0.5).unwrap_or_else(|| T::one() / T::from(2).unwrap())
                });

                if input.dim() == layer_output.dim() {
                    Ok(grad_output * gate)
                } else if let Some(ref mut projection) = self.projection {
                    let projected_grad = grad_output * gate;
                    projection.backward(&projected_grad)
                } else {
                    Ok(grad_output * gate)
                }
            }
        }
    }

    fn reset(&mut self) {
        self.last_input = None;
        self.last_layer_output = None;
        if let Some(ref mut projection) = self.projection {
            projection.reset();
        }
    }
}

/// Simple linear projection layer for dimension matching in residual connections.
#[derive(Debug, Clone)]
pub struct LinearProjection<T: FloatBounds> {
    /// Weight matrix
    weights: Array2<T>,
    /// Bias vector (optional)
    bias: Option<Array1<T>>,
    /// Last input for backward pass
    last_input: Option<Array2<T>>,
    /// Input size
    input_size: usize,
    /// Output size  
    output_size: usize,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> LinearProjection<T> {
    /// Create a new linear projection layer.
    pub fn new(input_size: usize, output_size: usize) -> NeuralResult<Self> {
        // Initialize weights with Xavier/Glorot initialization
        let scale = T::from(2.0).unwrap_or(T::one() + T::one())
            / T::from(input_size + output_size).unwrap_or(T::one());
        let bound = scale.sqrt();

        let mut weights = Array2::zeros((input_size, output_size));
        for elem in weights.iter_mut() {
            // Simple initialization - in practice would use proper random number generation
            *elem = bound * (T::from(0.1).unwrap_or(T::one() / T::from(10).unwrap()));
        }

        Ok(Self {
            weights,
            bias: Some(Array1::zeros(output_size)),
            last_input: None,
            input_size,
            output_size,
        })
    }
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> Layer<T> for LinearProjection<T> {
    fn forward(&mut self, input: &Array2<T>, _training: bool) -> NeuralResult<Array2<T>> {
        if input.dim().1 != self.input_size {
            return Err(SklearsError::InvalidInput(format!(
                "Input size {} doesn't match expected {}",
                input.dim().1,
                self.input_size
            )));
        }

        self.last_input = Some(input.clone());

        let mut output = input.dot(&self.weights);

        if let Some(ref bias) = self.bias {
            for mut row in output.axis_iter_mut(scirs2_core::ndarray::Axis(0)) {
                row += bias;
            }
        }

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Array2<T>) -> NeuralResult<Array2<T>> {
        let _input = self.last_input.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("No input stored for backward pass".to_string())
        })?;

        // Compute input gradient: grad_input = grad_output @ weights.T
        let grad_input = grad_output.dot(&self.weights.t());

        Ok(grad_input)
    }

    fn reset(&mut self) {
        self.last_input = None;
    }
}

// Import ndarray slice notation
use scirs2_core::ndarray::s;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_residual_addition() {
        let mut residual = ResidualBlock::<f64>::new(ResidualType::Addition, false);

        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let layer_output = array![[0.1, 0.2], [0.3, 0.4]];

        let result = residual.apply_residual(&input, &layer_output).unwrap();
        let expected = array![[1.1, 2.2], [3.3, 4.4]];

        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_residual_concatenation() {
        let mut residual = ResidualBlock::<f64>::new(ResidualType::Concatenation, false);

        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let layer_output = array![[0.1], [0.3]];

        let result = residual.apply_residual(&input, &layer_output).unwrap();
        let expected = array![[1.0, 2.0, 0.1], [3.0, 4.0, 0.3]];

        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_residual_gated() {
        let mut residual = ResidualBlock::<f64>::new(ResidualType::Gated, false);
        residual.set_gate(0.3).unwrap();

        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let layer_output = array![[0.1, 0.2], [0.3, 0.4]];

        let result = residual.apply_residual(&input, &layer_output).unwrap();

        // Expected: 0.3 * input + 0.7 * layer_output
        let expected = &input * 0.3 + &layer_output * 0.7;

        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_projection() {
        let mut projection = LinearProjection::<f64>::new(2, 3).unwrap();
        let input = array![[1.0, 2.0], [3.0, 4.0]];

        let output = projection.forward(&input, false).unwrap();

        // Check output shape
        assert_eq!(output.dim(), (2, 3));

        // Test backward pass
        let grad_output = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let grad_input = projection.backward(&grad_output).unwrap();

        // Check gradient shape
        assert_eq!(grad_input.dim(), (2, 2));
    }

    #[test]
    fn test_dimension_mismatch_handling() {
        let mut residual = ResidualBlock::<f64>::new(ResidualType::Addition, true);

        let input = array![[1.0, 2.0], [3.0, 4.0]]; // 2x2
        let layer_output = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]; // 2x3

        let result = residual.apply_residual(&input, &layer_output).unwrap();

        // Should automatically project input to match layer_output dimensions
        assert_eq!(result.dim(), (2, 3));
    }

    #[test]
    fn test_residual_backward() {
        let mut residual = ResidualBlock::<f64>::new(ResidualType::Addition, false);

        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let layer_output = array![[0.1, 0.2], [0.3, 0.4]];

        // Forward pass
        let _result = residual.apply_residual(&input, &layer_output).unwrap();

        // Backward pass
        let grad_output = array![[1.0, 1.0], [1.0, 1.0]];
        let grad_input = residual.backward(&grad_output).unwrap();

        // For addition, gradient should flow unchanged
        assert_abs_diff_eq!(grad_input, grad_output, epsilon = 1e-10);
    }

    #[test]
    fn test_residual_types_validation() {
        let mut residual = ResidualBlock::<f64>::new(ResidualType::Addition, false);

        // Should fail to set gate on non-gated residual
        assert!(residual.set_gate(0.5).is_err());

        // Gated residual should allow gate setting
        let mut gated_residual = ResidualBlock::<f64>::new(ResidualType::Gated, false);
        assert!(gated_residual.set_gate(0.3).is_ok());
        assert_abs_diff_eq!(gated_residual.get_gate().unwrap(), 0.3, epsilon = 1e-10);
    }
}
