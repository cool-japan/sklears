//! Layer Normalization implementation.
//!
//! Layer normalization normalizes across the feature dimension for each sample,
//! making it particularly suitable for transformer architectures and recurrent
//! neural networks. This implementation follows the paper "Layer Normalization"
//! by Ba, Kiros, and Hinton (2016).

use super::{Layer, LayerConfig, ParameterizedLayer};
use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Axis, ScalarOperand};
use scirs2_core::numeric::FromPrimitive;
use scirs2_core::random::Rng;
use sklears_core::{
    error::SklearsError,
    types::FloatBounds,
    validation::{ConfigValidation, Validate, ValidationRule, ValidationRules},
};
use std::marker::PhantomData;

/// Configuration for Layer Normalization
#[derive(Debug, Clone)]
pub struct LayerNormConfig<T: FloatBounds> {
    /// Number of features (input size)
    pub num_features: usize,
    /// Small constant added to variance for numerical stability
    pub epsilon: T,
    /// Whether to learn affine parameters (gamma and beta)
    pub affine: bool,
}

impl<T: FloatBounds> Default for LayerNormConfig<T> {
    fn default() -> Self {
        Self {
            num_features: 0,
            epsilon: T::from_f64(1e-5).unwrap_or_else(|| T::epsilon()),
            affine: true,
        }
    }
}

impl<T: FloatBounds> Validate for LayerNormConfig<T> {
    fn validate(&self) -> sklears_core::error::Result<()> {
        // Validate num_features
        ValidationRules::new("num_features")
            .add_rule(ValidationRule::Positive)
            .validate_numeric(&self.num_features)?;

        // Validate epsilon
        ValidationRules::new("epsilon")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.epsilon)?;

        Ok(())
    }
}

impl<T: FloatBounds> ConfigValidation for LayerNormConfig<T> {
    fn validate_config(&self) -> sklears_core::error::Result<()> {
        self.validate()?;

        let epsilon_f64 = self.epsilon.to_f64().unwrap_or(1e-5);
        if epsilon_f64 < 1e-8 {
            log::warn!(
                "Very small epsilon ({:.2e}) may cause numerical instability",
                epsilon_f64
            );
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if !self.affine {
            warnings.push("Disabled affine parameters may reduce model expressiveness".to_string());
        }

        warnings
    }
}

/// Layer Normalization
///
/// Applies layer normalization to incoming data:
///
/// For each sample in the batch:
/// - Computes mean and variance across features
/// - Normalizes: (x - mean) / sqrt(var + epsilon)
/// - Applies learnable affine transformation: gamma * normalized + beta (if affine=True)
///
/// Unlike batch normalization which normalizes across the batch dimension,
/// layer normalization normalizes across the feature dimension for each sample.
#[derive(Debug, Clone)]
pub struct LayerNorm<T: FloatBounds = f64> {
    config: LayerNormConfig<T>,

    // Learnable parameters (if affine=True)
    weight: Option<Array1<T>>, // gamma (scale)
    bias: Option<Array1<T>>,   // beta (shift)

    // Cached values for backward pass
    cached_input: Option<Array2<T>>,
    cached_normalized: Option<Array2<T>>,
    cached_mean: Option<Array1<T>>,
    cached_var: Option<Array1<T>>,
    cached_std: Option<Array1<T>>,

    // Gradients
    weight_grad: Option<Array1<T>>,
    bias_grad: Option<Array1<T>>,

    _phantom: PhantomData<T>,
}

impl<T: FloatBounds + ScalarOperand> LayerNorm<T> {
    /// Create a new Layer Normalization layer
    pub fn new(num_features: usize) -> Self {
        Self {
            config: LayerNormConfig {
                num_features,
                ..Default::default()
            },
            weight: None,
            bias: None,
            cached_input: None,
            cached_normalized: None,
            cached_mean: None,
            cached_var: None,
            cached_std: None,
            weight_grad: None,
            bias_grad: None,
            _phantom: PhantomData,
        }
    }

    /// Create a layer norm with custom configuration
    pub fn with_config(config: LayerNormConfig<T>) -> NeuralResult<Self> {
        config.validate_config()?;

        let mut layer = Self {
            config,
            weight: None,
            bias: None,
            cached_input: None,
            cached_normalized: None,
            cached_mean: None,
            cached_var: None,
            cached_std: None,
            weight_grad: None,
            bias_grad: None,
            _phantom: PhantomData,
        };

        layer.initialize();
        Ok(layer)
    }

    /// Builder pattern methods
    pub fn epsilon(mut self, epsilon: T) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.config.affine = affine;
        self
    }

    /// Initialize the layer parameters
    pub fn initialize(&mut self) {
        // Initialize learnable parameters
        if self.config.affine {
            self.weight = Some(Array1::ones(self.config.num_features));
            self.bias = Some(Array1::zeros(self.config.num_features));
            self.weight_grad = Some(Array1::zeros(self.config.num_features));
            self.bias_grad = Some(Array1::zeros(self.config.num_features));
        }
    }

    /// Reset parameters to default values
    pub fn reset_parameters(&mut self) {
        if self.config.affine {
            if let Some(ref mut weight) = self.weight {
                weight.fill(T::one());
            }
            if let Some(ref mut bias) = self.bias {
                bias.fill(T::zero());
            }
        }
    }

    /// Initialize parameters with random values
    pub fn initialize_parameters_random<R: Rng>(&mut self, rng: &mut R) {
        if self.config.affine {
            if let Some(ref mut weight) = self.weight {
                for w in weight.iter_mut() {
                    *w = T::from_f64(rng.gen()).unwrap_or(T::one());
                }
            }
            if let Some(ref mut bias) = self.bias {
                for b in bias.iter_mut() {
                    *b = T::from_f64(rng.gen::<f64>() - 0.5).unwrap_or(T::zero());
                }
            }
        }
    }

    /// Get weight (gamma) parameters
    pub fn weight(&self) -> Option<&Array1<T>> {
        self.weight.as_ref()
    }

    /// Get bias (beta) parameters
    pub fn bias(&self) -> Option<&Array1<T>> {
        self.bias.as_ref()
    }

    /// Get weight gradients
    pub fn weight_grad(&self) -> Option<&Array1<T>> {
        self.weight_grad.as_ref()
    }

    /// Get bias gradients
    pub fn bias_grad(&self) -> Option<&Array1<T>> {
        self.bias_grad.as_ref()
    }
}

impl<T: FloatBounds + ScalarOperand> Layer<T> for LayerNorm<T> {
    fn forward(&mut self, input: &Array2<T>, training: bool) -> NeuralResult<Array2<T>> {
        let batch_size = input.nrows();
        let num_features = input.ncols();

        if num_features != self.config.num_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.config.num_features, num_features
            )));
        }

        if batch_size == 0 {
            return Err(SklearsError::InvalidInput(
                "Batch size cannot be zero".to_string(),
            ));
        }

        // Initialize if not done yet
        if self.weight.is_none() && self.config.affine {
            self.initialize();
        }

        // Compute statistics across features for each sample
        let mean = input.mean_axis(Axis(1)).unwrap();

        // Compute variance across features for each sample: E[(X - μ)²]
        let centered = input - &mean.view().insert_axis(Axis(1));
        let var = centered.mapv(|x| x * x).mean_axis(Axis(1)).unwrap();

        // Normalize: (x - mean) / sqrt(var + epsilon)
        let std = var.mapv(|v| (v + self.config.epsilon).sqrt());
        let normalized = &centered / &std.view().insert_axis(Axis(1));

        // Apply affine transformation if enabled: gamma * normalized + beta
        let output = if self.config.affine {
            if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                &normalized * &weight.view().insert_axis(Axis(0))
                    + &bias.view().insert_axis(Axis(0))
            } else {
                normalized.clone()
            }
        } else {
            normalized.clone()
        };

        // Cache values for backward pass
        if training {
            self.cached_input = Some(input.clone());
            self.cached_normalized = Some(normalized);
            self.cached_mean = Some(mean);
            self.cached_var = Some(var);
            self.cached_std = Some(std);
        }

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Array2<T>) -> NeuralResult<Array2<T>> {
        let batch_size = grad_output.nrows();
        let num_features = grad_output.ncols();

        if num_features != self.config.num_features {
            return Err(SklearsError::InvalidInput(format!(
                "Gradient output features {} don't match layer features {}",
                num_features, self.config.num_features
            )));
        }

        // Get cached values from forward pass
        let input = self.cached_input.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("No cached input for backward pass".to_string())
        })?;
        let normalized = self
            .cached_normalized
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("No cached normalized values".to_string()))?;
        let mean = self
            .cached_mean
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("No cached mean".to_string()))?;
        let var = self
            .cached_var
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("No cached variance".to_string()))?;
        let std = self
            .cached_std
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("No cached std".to_string()))?;

        // Compute parameter gradients if affine
        if self.config.affine {
            if let (Some(ref mut weight_grad), Some(ref mut bias_grad)) =
                (&mut self.weight_grad, &mut self.bias_grad)
            {
                // Gradient w.r.t. gamma (weight): sum over batch of grad_out * normalized
                *weight_grad = (grad_output * normalized).sum_axis(Axis(0));

                // Gradient w.r.t. beta (bias): sum over batch of grad_out
                *bias_grad = grad_output.sum_axis(Axis(0));
            }
        }

        // Compute input gradients
        let grad_input = if self.config.affine {
            if let Some(ref weight) = self.weight {
                // Scale by gamma (weight)
                grad_output * &weight.view().insert_axis(Axis(0))
            } else {
                grad_output.clone()
            }
        } else {
            grad_output.clone()
        };

        // Backprop through layer normalization
        // For layer norm, gradients flow across the feature dimension within each sample
        let inv_std = std.mapv(|s| T::one() / s);
        let num_features_t = T::from_f64(num_features as f64).unwrap_or(T::one());

        // Gradient of normalized values
        let grad_norm = &grad_input;

        // Gradient w.r.t. variance for each sample
        let centered = input - &mean.view().insert_axis(Axis(1));
        let grad_var = (grad_norm * &centered).sum_axis(Axis(1))
            * FromPrimitive::from_f64(-0.5).unwrap_or(T::zero())
            * inv_std.mapv(|x| x * x * x);

        // Gradient w.r.t. mean for each sample
        let grad_mean = grad_norm.sum_axis(Axis(1))
            * FromPrimitive::from_f64(-1.0).unwrap_or(T::zero())
            * &inv_std
            + &grad_var
                * &centered.sum_axis(Axis(1))
                * FromPrimitive::from_f64(-2.0).unwrap_or(T::zero())
                / num_features_t;

        // Gradient w.r.t. input
        let grad_input_final = grad_norm * &inv_std.view().insert_axis(Axis(1))
            + &(&grad_var.view().insert_axis(Axis(1))
                * &centered
                * T::from_f64(2.0).unwrap_or(T::one()))
                / num_features_t
            + &grad_mean.view().insert_axis(Axis(1)) / num_features_t;

        Ok(grad_input_final)
    }

    fn num_parameters(&self) -> usize {
        if self.config.affine {
            2 * self.config.num_features // gamma + beta
        } else {
            0
        }
    }

    fn reset(&mut self) {
        self.cached_input = None;
        self.cached_normalized = None;
        self.cached_mean = None;
        self.cached_var = None;
        self.cached_std = None;

        if let Some(ref mut weight_grad) = self.weight_grad {
            weight_grad.fill(T::zero());
        }
        if let Some(ref mut bias_grad) = self.bias_grad {
            bias_grad.fill(T::zero());
        }
    }
}

impl<T: FloatBounds + ScalarOperand> ParameterizedLayer<T> for LayerNorm<T> {
    fn parameters(&self) -> Vec<&Array1<T>> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array1<T>> {
        let mut params = Vec::new();
        if let Some(ref mut weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameter_gradients(&self) -> Vec<Array1<T>> {
        let mut grads = Vec::new();
        if let Some(ref weight_grad) = self.weight_grad {
            grads.push(weight_grad.clone());
        }
        if let Some(ref bias_grad) = self.bias_grad {
            grads.push(bias_grad.clone());
        }
        grads
    }

    fn update_parameters(&mut self, updates: &[Array1<T>]) -> NeuralResult<()> {
        if !self.config.affine {
            return Ok(());
        }

        let expected_updates = if self.weight.is_some() && self.bias.is_some() {
            2
        } else if self.weight.is_some() || self.bias.is_some() {
            1
        } else {
            0
        };

        if updates.len() != expected_updates {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} parameter updates, got {}",
                expected_updates,
                updates.len()
            )));
        }

        let mut update_idx = 0;

        if let Some(ref mut weight) = self.weight {
            *weight = &*weight + &updates[update_idx];
            update_idx += 1;
        }

        if let Some(ref mut bias) = self.bias {
            *bias = &*bias + &updates[update_idx];
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    /// Helper to compare arrays element-by-element since approx doesn't implement AbsDiffEq for Array
    fn assert_arrays_close<D: scirs2_core::ndarray::Dimension>(
        a: &scirs2_core::ndarray::Array<f64, D>,
        b: &scirs2_core::ndarray::Array<f64, D>,
        epsilon: f64,
    ) {
        assert_eq!(a.shape(), b.shape(), "Array shapes differ");
        for (av, bv) in a.iter().zip(b.iter()) {
            assert_abs_diff_eq!(*av, *bv, epsilon = epsilon);
        }
    }

    #[test]
    #[ignore]
    fn test_layer_norm_forward() {
        let mut ln = LayerNorm::new(3);
        ln.initialize();

        // Simple input batch: 2 samples, 3 features
        let input = array![[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]];

        let output = ln.forward(&input, true).unwrap();

        // Check output shape
        assert_eq!(output.dim(), (2, 3));

        // For layer norm, each row should be normalized independently
        // Check that output has approximately zero mean and unit variance along features
        for row in output.axis_iter(Axis(0)) {
            let mean = row.mean().unwrap();
            let var = row.mapv(|x| (x - mean) * (x - mean)).mean().unwrap();

            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(var, 1.0, epsilon = 1e-4);
        }
    }

    #[test]
    #[ignore]
    fn test_layer_norm_without_affine() {
        let mut ln = LayerNorm::new(3).affine(false);
        ln.initialize();

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let output = ln.forward(&input, true).unwrap();

        // Without affine, each row should have zero mean and unit variance
        for (i, row) in output.axis_iter(Axis(0)).enumerate() {
            let mean = row.mean().unwrap();
            let var = row.mapv(|x| (x - mean) * (x - mean)).mean().unwrap();

            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(var, 1.0, epsilon = 1e-4);
        }
    }

    #[test]
    #[ignore]
    fn test_layer_norm_backward() {
        let mut ln = LayerNorm::new(3);
        ln.initialize();

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Forward pass
        let output = ln.forward(&input, true).unwrap();

        // Backward pass with unit gradients
        let grad_output = Array2::ones((2, 3));
        let grad_input = ln.backward(&grad_output).unwrap();

        // Check gradient shape
        assert_eq!(grad_input.dim(), input.dim());

        // For layer norm, the sum of input gradients along features should be zero
        // (due to the mean centering in the normalization)
        for row in grad_input.axis_iter(Axis(0)) {
            let grad_sum = row.sum();
            assert_abs_diff_eq!(grad_sum, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    #[ignore]
    fn test_layer_norm_parameter_gradients() {
        let mut ln = LayerNorm::new(2);
        ln.initialize();

        let input = array![[1.0, 2.0], [3.0, 4.0]];

        // Forward pass
        ln.forward(&input, true).unwrap();

        // Backward pass
        let grad_output = array![[1.0, 2.0], [3.0, 4.0]];
        ln.backward(&grad_output).unwrap();

        // Check parameter gradients exist
        assert!(ln.weight_grad.is_some());
        assert!(ln.bias_grad.is_some());

        let weight_grad = ln.weight_grad().unwrap();
        let bias_grad = ln.bias_grad().unwrap();

        // Bias gradient should be sum of grad_output
        let expected_bias_grad = grad_output.sum_axis(Axis(0));
        // Compare element by element
        for (a, b) in bias_grad.iter().zip(expected_bias_grad.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    #[test]
    #[ignore]
    fn test_layer_norm_config_validation() {
        let config = LayerNormConfig {
            num_features: 0, // Invalid
            epsilon: 1e-5,
            affine: true,
        };

        assert!(config.validate().is_err());

        let valid_config = LayerNormConfig {
            num_features: 10,
            epsilon: 1e-5,
            affine: true,
        };

        assert!(valid_config.validate().is_ok());
    }

    #[test]
    #[ignore]
    fn test_layer_norm_reset() {
        let mut ln = LayerNorm::new(2);
        ln.initialize();

        let input = array![[1.0, 2.0], [3.0, 4.0]];
        ln.forward(&input, true).unwrap();

        // Check that cache is populated
        assert!(ln.cached_input.is_some());

        // Reset
        ln.reset();

        // Check that cache is cleared
        assert!(ln.cached_input.is_none());
        assert!(ln.cached_mean.is_none());
    }

    #[test]
    #[ignore]
    fn test_layer_norm_vs_manual_computation() {
        let mut ln = LayerNorm::new(3).epsilon(1e-8).affine(false); // Disable affine for pure normalization
        ln.initialize();

        let input = array![[1.0, 4.0, 7.0], [2.0, 5.0, 8.0]];

        let output = ln.forward(&input, false).unwrap();

        // Manual computation for first row
        let row1 = input.row(0);
        let mean1 = row1.mean().unwrap();
        let var1 = row1.mapv(|x| (x - mean1) * (x - mean1)).mean().unwrap();
        let std1 = (var1 + 1e-8_f64).sqrt();
        let normalized1 = row1.mapv(|x| (x - mean1) / std1);

        // Compare element by element
        for (a, b) in output.row(0).iter().zip(normalized1.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }
    }
}
