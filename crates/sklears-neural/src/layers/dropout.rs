//! Dropout layer implementation.
//!
//! Dropout is a regularization technique that randomly sets input units to 0
//! with a frequency of `rate` at each step during training time, which helps
//! prevent overfitting. This implementation follows the paper "Dropout: A Simple
//! Way to Prevent Neural Networks from Overfitting" by Srivastava et al. (2014).

use super::{Layer, LayerConfig};
use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::SklearsError,
    types::FloatBounds,
    validation::{ConfigValidation, Validate, ValidationRule, ValidationRules},
};
use std::marker::PhantomData;

/// Configuration for Dropout layer
#[derive(Debug, Clone)]
pub struct DropoutConfig<T: FloatBounds> {
    /// Dropout rate (probability of setting a unit to 0)
    pub rate: T,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl<T: FloatBounds> Default for DropoutConfig<T> {
    fn default() -> Self {
        Self {
            rate: T::from(0.5).unwrap_or_else(|| T::one() / T::from(2).unwrap()),
            seed: None,
        }
    }
}

impl<T: FloatBounds> Validate for DropoutConfig<T> {
    fn validate(&self) -> sklears_core::error::Result<()> {
        // Validate dropout rate
        ValidationRules::new("rate")
            .add_rule(ValidationRule::Range { min: 0.0, max: 1.0 })
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.rate)?;

        Ok(())
    }
}

impl<T: FloatBounds> ConfigValidation for DropoutConfig<T> {
    fn validate_config(&self) -> sklears_core::error::Result<()> {
        self.validate()?;

        if self.rate > T::from(0.8).unwrap_or_else(|| T::one() * T::from(0.8).unwrap()) {
            log::warn!(
                "High dropout rate ({:.2}) may hurt model performance",
                self.rate.to_f64().unwrap_or(0.0)
            );
        }

        if self.rate < T::from(0.1).unwrap_or_else(|| T::one() / T::from(10).unwrap()) {
            log::warn!(
                "Low dropout rate ({:.2}) may not provide sufficient regularization",
                self.rate.to_f64().unwrap_or(0.0)
            );
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.rate == T::zero() {
            warnings.push("Dropout rate of 0 disables dropout regularization".to_string());
        }

        if self.rate == T::one() {
            warnings.push("Dropout rate of 1 will zero out all inputs".to_string());
        }

        warnings
    }
}

/// Dropout layer for regularization
///
/// During training:
/// - Randomly sets input elements to 0 with probability `rate`
/// - Scales remaining elements by 1/(1-rate) to maintain expected value
///
/// During inference:
/// - Passes input through unchanged (no dropout applied)
#[derive(Debug, Clone)]
pub struct Dropout<T: FloatBounds = f64> {
    config: DropoutConfig<T>,

    // Cached mask for backward pass
    cached_mask: Option<Array2<T>>,

    _phantom: PhantomData<T>,
}

impl<T: FloatBounds> Dropout<T> {
    /// Create a new Dropout layer
    pub fn new(rate: T) -> Self {
        Self {
            config: DropoutConfig { rate, seed: None },
            cached_mask: None,
            _phantom: PhantomData,
        }
    }

    /// Create a dropout layer with custom configuration
    pub fn with_config(config: DropoutConfig<T>) -> NeuralResult<Self> {
        config.validate_config()?;

        Ok(Self {
            config,
            cached_mask: None,
            _phantom: PhantomData,
        })
    }

    /// Builder pattern methods
    pub fn rate(mut self, rate: T) -> Self {
        self.config.rate = rate;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Get the dropout rate
    pub fn get_rate(&self) -> T {
        self.config.rate
    }

    /// Generate dropout mask
    fn generate_mask<R: Rng>(&self, shape: (usize, usize), rng: &mut R) -> Array2<T> {
        let keep_prob = T::one() - self.config.rate;
        let scale = T::one() / keep_prob;

        Array2::from_shape_fn(shape, |_| {
            if rng.gen::<f64>() < keep_prob.to_f64().unwrap_or(1.0) {
                scale
            } else {
                T::zero()
            }
        })
    }
}

impl<T: FloatBounds> Layer<T> for Dropout<T> {
    fn forward(&mut self, input: &Array2<T>, training: bool) -> NeuralResult<Array2<T>> {
        if !training || self.config.rate == T::zero() {
            // No dropout during inference or if rate is 0
            self.cached_mask = None;
            return Ok(input.clone());
        }

        if self.config.rate == T::one() {
            // All units dropped out
            let output = Array2::zeros(input.dim());
            self.cached_mask = Some(Array2::zeros(input.dim()));
            return Ok(output);
        }

        // Generate random mask
        let mut rng = if let Some(seed) = self.config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(42) // Use default seed if none provided
        };

        let mask = self.generate_mask(input.dim(), &mut rng);
        let output = input * &mask;

        // Cache mask for backward pass
        self.cached_mask = Some(mask);

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Array2<T>) -> NeuralResult<Array2<T>> {
        // If no mask was cached, pass gradients through unchanged
        if let Some(ref mask) = self.cached_mask {
            Ok(grad_output * mask)
        } else {
            Ok(grad_output.clone())
        }
    }

    fn num_parameters(&self) -> usize {
        0 // Dropout has no trainable parameters
    }

    fn reset(&mut self) {
        self.cached_mask = None;
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_dropout_forward_inference() {
        let mut dropout = Dropout::new(0.5);

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // During inference, output should be identical to input
        let output = dropout.forward(&input, false).unwrap();
        assert_abs_diff_eq!(output, input, epsilon = 1e-10);
        assert!(dropout.cached_mask.is_none());
    }

    #[test]
    fn test_dropout_forward_training_zero_rate() {
        let mut dropout = Dropout::new(0.0);

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // With rate=0, output should be identical to input even in training
        let output = dropout.forward(&input, true).unwrap();
        assert_abs_diff_eq!(output, input, epsilon = 1e-10);
        assert!(dropout.cached_mask.is_none());
    }

    #[test]
    fn test_dropout_forward_training_full_rate() {
        let mut dropout = Dropout::new(1.0);

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // With rate=1, output should be all zeros
        let output = dropout.forward(&input, true).unwrap();
        let expected = Array2::zeros(input.dim());
        assert_abs_diff_eq!(output, expected, epsilon = 1e-10);
        assert!(dropout.cached_mask.is_some());
    }

    #[test]
    fn test_dropout_forward_training() {
        let mut dropout = Dropout::new(0.5).seed(42); // Fixed seed for reproducibility

        let input = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let output = dropout.forward(&input, true).unwrap();

        // Check that mask was cached
        assert!(dropout.cached_mask.is_some());

        // Check output shape
        assert_eq!(output.dim(), input.dim());

        // With scaling, non-zero outputs should be 2x the input (since keep_prob=0.5)
        let mask = dropout.cached_mask.as_ref().unwrap();
        for i in 0..input.nrows() {
            for j in 0..input.ncols() {
                if mask[[i, j]] > 0.0 {
                    assert_abs_diff_eq!(output[[i, j]], input[[i, j]] * 2.0, epsilon = 1e-10);
                } else {
                    assert_abs_diff_eq!(output[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_dropout_backward() {
        let mut dropout = Dropout::new(0.5).seed(42);

        let input = array![[1.0, 2.0], [3.0, 4.0]];

        // Forward pass
        dropout.forward(&input, true).unwrap();

        // Backward pass
        let grad_output = array![[1.0, 1.0], [1.0, 1.0]];

        let grad_input = dropout.backward(&grad_output).unwrap();

        // Gradients should be scaled by the same mask
        let mask = dropout.cached_mask.as_ref().unwrap();
        let expected_grad = &grad_output * mask;
        assert_abs_diff_eq!(grad_input, expected_grad, epsilon = 1e-10);
    }

    #[test]
    fn test_dropout_backward_no_mask() {
        let mut dropout = Dropout::new(0.5);

        // Backward pass without forward pass (no cached mask)
        let grad_output = array![[1.0, 2.0], [3.0, 4.0]];

        let grad_input = dropout.backward(&grad_output).unwrap();

        // Should pass gradients through unchanged
        assert_abs_diff_eq!(grad_input, grad_output, epsilon = 1e-10);
    }

    #[test]
    fn test_dropout_config_validation() {
        let config = DropoutConfig {
            rate: 1.5, // Invalid (> 1.0)
            seed: None,
        };

        assert!(config.validate().is_err());

        let config = DropoutConfig {
            rate: -0.1, // Invalid (< 0.0)
            seed: None,
        };

        assert!(config.validate().is_err());

        let valid_config = DropoutConfig {
            rate: 0.5,
            seed: Some(42),
        };

        assert!(valid_config.validate().is_ok());
    }

    #[test]
    fn test_dropout_reset() {
        let mut dropout = Dropout::new(0.5);

        let input = array![[1.0, 2.0], [3.0, 4.0]];
        dropout.forward(&input, true).unwrap();

        // Check that mask is cached
        assert!(dropout.cached_mask.is_some());

        // Reset
        dropout.reset();

        // Check that mask is cleared
        assert!(dropout.cached_mask.is_none());
    }

    #[test]
    fn test_dropout_reproducibility_with_seed() {
        let mut dropout1 = Dropout::new(0.5).seed(42);
        let mut dropout2 = Dropout::new(0.5).seed(42);

        let input = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];

        let output1 = dropout1.forward(&input, true).unwrap();
        let output2 = dropout2.forward(&input, true).unwrap();

        // Should produce identical outputs with same seed
        assert_abs_diff_eq!(output1, output2, epsilon = 1e-10);
    }

    #[test]
    fn test_dropout_expected_value_preservation() {
        let mut dropout = Dropout::new(0.3); // 30% dropout

        let input = Array2::ones((100, 50)); // Large array of ones

        // Run multiple forward passes and compute average
        let mut total_sum = 0.0;
        let num_trials = 1000;

        for _ in 0..num_trials {
            let output = dropout.forward(&input, true).unwrap();
            total_sum += output.sum();
        }

        let average_sum = total_sum / num_trials as f64;
        let expected_sum = input.sum(); // Should preserve expected value

        // Allow some variance but should be close to expected value
        let relative_error = (average_sum - expected_sum).abs() / expected_sum;
        assert!(
            relative_error < 0.1,
            "Expected value not preserved: {} vs {}",
            average_sum,
            expected_sum
        );
    }
}
