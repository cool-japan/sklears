//! Batch Normalization layer implementation.
//!
//! Batch normalization helps stabilize and accelerate neural network training by
//! normalizing layer inputs to have zero mean and unit variance. This implementation
//! follows the paper "Batch Normalization: Accelerating Deep Network Training by
//! Reducing Internal Covariate Shift" by Ioffe and Szegedy (2015).

use super::{Layer, LayerConfig, ParameterizedLayer};
use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Axis, ScalarOperand, Zip};
use scirs2_core::random::Rng;
use sklears_core::{
    error::SklearsError,
    types::FloatBounds,
    validation::{ConfigValidation, Validate, ValidationRule, ValidationRules},
};
use std::marker::PhantomData;

/// Configuration for Batch Normalization layer
#[derive(Debug, Clone)]
pub struct BatchNormConfig<T: FloatBounds> {
    /// Number of features (input size)
    pub num_features: usize,
    /// Momentum for moving average computation
    pub momentum: T,
    /// Small constant added to variance for numerical stability
    pub epsilon: T,
    /// Whether to learn affine parameters (gamma and beta)
    pub affine: bool,
    /// Whether to track running statistics (mean and variance)
    pub track_running_stats: bool,
}

impl<T: FloatBounds> Default for BatchNormConfig<T> {
    fn default() -> Self {
        Self {
            num_features: 0,
            momentum: T::from(0.1).unwrap_or_else(|| T::one() / T::from(10).unwrap()),
            epsilon: T::from(1e-5).unwrap_or_else(|| T::one() / T::from(100000).unwrap()),
            affine: true,
            track_running_stats: true,
        }
    }
}

impl<T: FloatBounds> Validate for BatchNormConfig<T> {
    fn validate(&self) -> sklears_core::error::Result<()> {
        // Validate num_features
        ValidationRules::new("num_features")
            .add_rule(ValidationRule::Positive)
            .validate_numeric(&self.num_features)?;

        // Validate momentum
        ValidationRules::new("momentum")
            .add_rule(ValidationRule::Range { min: 0.0, max: 1.0 })
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.momentum)?;

        // Validate epsilon
        ValidationRules::new("epsilon")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.epsilon)?;

        Ok(())
    }
}

impl<T: FloatBounds> ConfigValidation for BatchNormConfig<T> {
    fn validate_config(&self) -> sklears_core::error::Result<()> {
        self.validate()?;

        if self.momentum > T::from(0.5).unwrap_or_else(|| T::one() / T::from(2).unwrap()) {
            log::warn!(
                "High momentum ({:.3}) may lead to slow adaptation of running statistics",
                self.momentum.to_f64().unwrap_or(0.0)
            );
        }

        if self.epsilon < T::from(1e-8).unwrap_or_else(|| T::one() / T::from(100000000).unwrap()) {
            log::warn!(
                "Very small epsilon ({:.2e}) may cause numerical instability",
                self.epsilon.to_f64().unwrap_or(0.0)
            );
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if !self.affine {
            warnings.push("Disabled affine parameters may reduce model expressiveness".to_string());
        }

        if !self.track_running_stats {
            warnings.push(
                "Disabled running statistics tracking may affect inference behavior".to_string(),
            );
        }

        warnings
    }
}

/// Batch Normalization layer
///
/// Applies batch normalization to incoming data:
///
/// During training:
/// - Computes batch mean and variance
/// - Normalizes inputs: (x - mean) / sqrt(var + epsilon)
/// - Applies learnable affine transformation: gamma * normalized + beta
/// - Updates running statistics using exponential moving average
///
/// During inference:
/// - Uses running mean and variance for normalization
/// - Applies the same affine transformation
#[derive(Debug, Clone)]
pub struct BatchNorm1d<T: FloatBounds = f64> {
    config: BatchNormConfig<T>,

    // Learnable parameters (if affine=True)
    weight: Option<Array1<T>>, // gamma (scale)
    bias: Option<Array1<T>>,   // beta (shift)

    // Running statistics (if track_running_stats=True)
    running_mean: Option<Array1<T>>,
    running_var: Option<Array1<T>>,
    num_batches_tracked: usize,

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

impl<T: FloatBounds + ScalarOperand> BatchNorm1d<T> {
    /// Create a new Batch Normalization layer
    pub fn new(num_features: usize) -> Self {
        Self {
            config: BatchNormConfig {
                num_features,
                ..Default::default()
            },
            weight: None,
            bias: None,
            running_mean: None,
            running_var: None,
            num_batches_tracked: 0,
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

    /// Create a batch norm layer with custom configuration
    pub fn with_config(config: BatchNormConfig<T>) -> NeuralResult<Self> {
        config.validate_config()?;

        let mut layer = Self {
            config,
            weight: None,
            bias: None,
            running_mean: None,
            running_var: None,
            num_batches_tracked: 0,
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
    pub fn momentum(mut self, momentum: T) -> Self {
        self.config.momentum = momentum;
        self
    }

    pub fn epsilon(mut self, epsilon: T) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.config.affine = affine;
        self
    }

    pub fn track_running_stats(mut self, track: bool) -> Self {
        self.config.track_running_stats = track;
        self
    }

    /// Initialize the layer parameters and statistics
    pub fn initialize(&mut self) {
        // Initialize learnable parameters
        if self.config.affine {
            self.weight = Some(Array1::ones(self.config.num_features));
            self.bias = Some(Array1::zeros(self.config.num_features));
            self.weight_grad = Some(Array1::zeros(self.config.num_features));
            self.bias_grad = Some(Array1::zeros(self.config.num_features));
        }

        // Initialize running statistics
        if self.config.track_running_stats {
            self.running_mean = Some(Array1::zeros(self.config.num_features));
            self.running_var = Some(Array1::ones(self.config.num_features));
            self.num_batches_tracked = 0;
        }
    }

    /// Get running mean (for inspection)
    pub fn running_mean(&self) -> Option<&Array1<T>> {
        self.running_mean.as_ref()
    }

    /// Get running variance (for inspection)
    pub fn running_var(&self) -> Option<&Array1<T>> {
        self.running_var.as_ref()
    }

    /// Get number of batches tracked
    pub fn num_batches_tracked(&self) -> usize {
        self.num_batches_tracked
    }

    /// Reset running statistics
    pub fn reset_running_stats(&mut self) {
        if self.config.track_running_stats {
            if let Some(ref mut mean) = self.running_mean {
                mean.fill(T::zero());
            }
            if let Some(ref mut var) = self.running_var {
                var.fill(T::one());
            }
            self.num_batches_tracked = 0;
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
                    *w = T::from(rng.gen::<f64>()).unwrap_or(T::one());
                }
            }
            if let Some(ref mut bias) = self.bias {
                for b in bias.iter_mut() {
                    *b = T::from(rng.gen::<f64>() - 0.5).unwrap_or(T::zero());
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

impl<T: FloatBounds + ScalarOperand> Layer<T> for BatchNorm1d<T> {
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

        let (mean, var) = if training {
            // Compute batch statistics
            let batch_mean = input.mean_axis(Axis(0)).unwrap();

            // Compute batch variance: E[(X - μ)²]
            let centered = input - &batch_mean.view().insert_axis(Axis(0));
            let batch_var = centered.mapv(|x| x * x).mean_axis(Axis(0)).unwrap();

            // Update running statistics if tracking
            if self.config.track_running_stats {
                self.num_batches_tracked += 1;

                // Exponential moving average
                if let Some(ref mut running_mean) = self.running_mean {
                    let momentum_comp = T::one() - self.config.momentum;
                    *running_mean =
                        &*running_mean * momentum_comp + &batch_mean * self.config.momentum;
                }

                if let Some(ref mut running_var) = self.running_var {
                    // Use unbiased variance estimate for running statistics
                    let unbiased_var = &batch_var * T::from(batch_size).unwrap()
                        / T::from(batch_size - 1).unwrap().max(T::one());
                    let momentum_comp = T::one() - self.config.momentum;
                    *running_var =
                        &*running_var * momentum_comp + &unbiased_var * self.config.momentum;
                }
            }

            (batch_mean, batch_var)
        } else {
            // Use running statistics for inference
            if let (Some(ref mean), Some(ref var)) = (&self.running_mean, &self.running_var) {
                (mean.clone(), var.clone())
            } else {
                // Fallback to batch statistics if running stats not available
                let batch_mean = input.mean_axis(Axis(0)).unwrap();
                let centered = input - &batch_mean.view().insert_axis(Axis(0));
                let batch_var = centered.mapv(|x| x * x).mean_axis(Axis(0)).unwrap();
                (batch_mean, batch_var)
            }
        };

        // Normalize: (x - mean) / sqrt(var + epsilon)
        let std = var.mapv(|v| (v + self.config.epsilon).sqrt());
        let normalized =
            (input - &mean.view().insert_axis(Axis(0))) / &std.view().insert_axis(Axis(0));

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

        // Backprop through normalization
        let inv_std = std.mapv(|s| T::one() / s);
        let batch_size_t = T::from(batch_size).unwrap();

        // Gradient of normalized values
        let grad_norm = &grad_input;

        // Gradient w.r.t. variance
        let centered = input - &mean.view().insert_axis(Axis(0));
        let grad_var = (grad_norm * &centered).sum_axis(Axis(0))
            * T::from(-0.5).unwrap()
            * inv_std.mapv(|x| x * x * x);

        // Gradient w.r.t. mean
        let grad_mean = grad_norm.sum_axis(Axis(0)) * T::from(-1.0).unwrap() * &inv_std
            + &grad_var * &centered.sum_axis(Axis(0)) * T::from(-2.0).unwrap() / batch_size_t;

        // Gradient w.r.t. input
        let grad_input_final = grad_norm * &inv_std.view().insert_axis(Axis(0))
            + &(&grad_var.view().insert_axis(Axis(0)) * &centered * T::from(2.0).unwrap())
                / batch_size_t
            + &grad_mean.view().insert_axis(Axis(0)) / batch_size_t;

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

impl<T: FloatBounds + ScalarOperand> ParameterizedLayer<T> for BatchNorm1d<T> {
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
    fn test_batch_norm_forward_training() {
        let mut bn = BatchNorm1d::new(3);
        bn.initialize();

        // Simple input batch: 2 samples, 3 features
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let output = bn.forward(&input, true).unwrap();

        // Check output shape
        assert_eq!(output.dim(), (2, 3));

        // Check that batch statistics were computed
        assert!(bn.cached_mean.is_some());
        assert!(bn.cached_var.is_some());

        // Check running statistics were updated
        assert!(bn.running_mean.is_some());
        assert!(bn.running_var.is_some());
        assert_eq!(bn.num_batches_tracked, 1);
    }

    #[test]
    #[ignore]
    fn test_batch_norm_forward_inference() {
        let mut bn = BatchNorm1d::new(2);
        bn.initialize();

        // Set known running statistics
        bn.running_mean = Some(array![1.0, 2.0]);
        bn.running_var = Some(array![1.0, 4.0]);

        let input = array![
            [2.0, 6.0],  // (2-1)/1 = 1, (6-2)/2 = 2
            [0.0, -2.0]  // (0-1)/1 = -1, (-2-2)/2 = -2
        ];

        let output = bn.forward(&input, false).unwrap();

        // Expected normalized values (approximately)
        let expected = array![[1.0, 2.0], [-1.0, -2.0]];

        assert_arrays_close(&output, &expected, 1e-4);
    }

    #[test]
    #[ignore]
    fn test_batch_norm_without_affine() {
        let mut bn = BatchNorm1d::new(2).affine(false);
        bn.initialize();

        let input = array![[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]];

        let output = bn.forward(&input, true).unwrap();

        // Without affine, output should be normalized to ~N(0,1)
        let output_mean = output.mean_axis(Axis(0)).unwrap();
        let output_var = output.mapv(|x| x * x).mean_axis(Axis(0)).unwrap();

        assert_abs_diff_eq!(output_mean[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output_mean[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output_var[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(output_var[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    #[ignore]
    fn test_batch_norm_backward() {
        let mut bn = BatchNorm1d::new(2);
        bn.initialize();

        let input = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        // Forward pass
        let output = bn.forward(&input, true).unwrap();

        // Backward pass with unit gradients
        let grad_output = Array2::ones((3, 2));
        let grad_input = bn.backward(&grad_output).unwrap();

        // Check gradient shape
        assert_eq!(grad_input.dim(), input.dim());

        // For batch norm, the sum of input gradients should be zero
        // (due to the mean centering in the normalization)
        let grad_sum = grad_input.sum_axis(Axis(0));
        assert_abs_diff_eq!(grad_sum[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad_sum[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    #[ignore]
    fn test_batch_norm_parameter_gradients() {
        let mut bn = BatchNorm1d::new(2);
        bn.initialize();

        let input = array![[1.0, 2.0], [3.0, 4.0]];

        // Forward pass
        bn.forward(&input, true).unwrap();

        // Backward pass
        let grad_output = array![[1.0, 2.0], [3.0, 4.0]];
        bn.backward(&grad_output).unwrap();

        // Check parameter gradients exist
        assert!(bn.weight_grad.is_some());
        assert!(bn.bias_grad.is_some());

        let weight_grad = bn.weight_grad().unwrap();
        let bias_grad = bn.bias_grad().unwrap();

        // Bias gradient should be sum of grad_output
        let expected_bias_grad = grad_output.sum_axis(Axis(0));
        // Compare element by element
        for (a, b) in bias_grad.iter().zip(expected_bias_grad.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    #[test]
    #[ignore]
    fn test_batch_norm_config_validation() {
        let config = BatchNormConfig {
            num_features: 0, // Invalid
            momentum: 0.1,
            epsilon: 1e-5,
            affine: true,
            track_running_stats: true,
        };

        assert!(config.validate().is_err());

        let valid_config = BatchNormConfig {
            num_features: 10,
            momentum: 0.1,
            epsilon: 1e-5,
            affine: true,
            track_running_stats: true,
        };

        assert!(valid_config.validate().is_ok());
    }

    #[test]
    #[ignore]
    fn test_batch_norm_reset() {
        let mut bn = BatchNorm1d::new(2);
        bn.initialize();

        let input = array![[1.0, 2.0], [3.0, 4.0]];
        bn.forward(&input, true).unwrap();

        // Check that cache is populated
        assert!(bn.cached_input.is_some());

        // Reset
        bn.reset();

        // Check that cache is cleared
        assert!(bn.cached_input.is_none());
        assert!(bn.cached_mean.is_none());
    }

    #[test]
    #[ignore]
    fn test_batch_norm_running_stats_tracking() {
        let mut bn = BatchNorm1d::new(2);
        bn.initialize();

        let input1 = array![[0.0, 0.0], [1.0, 1.0]];
        let input2 = array![[2.0, 2.0], [3.0, 3.0]];

        // First batch
        bn.forward(&input1, true).unwrap();
        let mean1 = bn.running_mean().unwrap().clone();

        // Second batch
        bn.forward(&input2, true).unwrap();
        let mean2 = bn.running_mean().unwrap().clone();

        // Running mean should have changed (compare element by element)
        let means_changed = mean1.iter().zip(mean2.iter()).any(|(&a, &b): (&f64, &f64)| (a - b).abs() > 1e-10);
        assert!(means_changed);

        // Should have tracked 2 batches
        assert_eq!(bn.num_batches_tracked(), 2);
    }
}
