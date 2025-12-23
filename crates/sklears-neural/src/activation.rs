//! Activation functions for neural networks.

use scirs2_core::ndarray::{Array1, Array2, Zip};
use std::f64::consts::{PI, SQRT_2};

/// Activation functions for neural network layers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Activation {
    /// Identity activation: f(x) = x
    Identity,
    /// Logistic sigmoid: f(x) = 1 / (1 + exp(-x))
    Logistic,
    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh,
    /// Rectified Linear Unit: f(x) = max(0, x)
    #[default]
    Relu,
    /// Exponential Linear Unit: f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    Elu,
    /// Swish/SiLU activation: f(x) = x * sigmoid(x)
    Swish,
    /// Gaussian Error Linear Unit: f(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    Gelu,
    /// Mish activation: f(x) = x * tanh(softplus(x))
    Mish,
    /// Leaky ReLU: f(x) = max(alpha * x, x) where alpha is small (0.01)
    LeakyRelu,
    /// Parametric ReLU: f(x) = max(alpha * x, x) where alpha is learnable
    /// Note: This requires separate handling as a parameterized layer
    PRelu,
}

impl Activation {
    /// Apply the activation function to a 2D array (for hidden layers)
    pub fn apply(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Identity => x.clone(),
            Activation::Logistic => x.mapv(|val| 1.0 / (1.0 + (-val).exp())),
            Activation::Tanh => x.mapv(|val| val.tanh()),
            Activation::Relu => x.mapv(|val| val.max(0.0)),
            Activation::Elu => x.mapv(|val| {
                if val > 0.0 {
                    val
                } else {
                    1.0 * (val.exp() - 1.0)
                }
            }),
            Activation::Swish => x.mapv(|val| val * (1.0 / (1.0 + (-val).exp()))),
            Activation::Gelu => x.mapv(|val| 0.5 * val * (1.0 + erf_approx(val / SQRT_2))),
            Activation::Mish => x.mapv(|val| val * softplus(val).tanh()),
            Activation::LeakyRelu => x.mapv(|val| if val > 0.0 { val } else { 0.01 * val }),
            Activation::PRelu => {
                // Note: PReLU should be used as a separate parameterized layer
                // This is a fallback with alpha=0.25 (similar to default initialization)
                x.mapv(|val| if val > 0.0 { val } else { 0.25 * val })
            }
        }
    }

    /// Apply the activation function to a 1D array (for output layer)
    pub fn apply_1d(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::Identity => x.clone(),
            Activation::Logistic => x.mapv(|val| 1.0 / (1.0 + (-val).exp())),
            Activation::Tanh => x.mapv(|val| val.tanh()),
            Activation::Relu => x.mapv(|val| val.max(0.0)),
            Activation::Elu => x.mapv(|val| {
                if val > 0.0 {
                    val
                } else {
                    1.0 * (val.exp() - 1.0)
                }
            }),
            Activation::Swish => x.mapv(|val| val * (1.0 / (1.0 + (-val).exp()))),
            Activation::Gelu => x.mapv(|val| 0.5 * val * (1.0 + erf_approx(val / SQRT_2))),
            Activation::Mish => x.mapv(|val| val * softplus(val).tanh()),
            Activation::LeakyRelu => x.mapv(|val| if val > 0.0 { val } else { 0.01 * val }),
            Activation::PRelu => {
                // Note: PReLU should be used as a separate parameterized layer
                // This is a fallback with alpha=0.25 (similar to default initialization)
                x.mapv(|val| if val > 0.0 { val } else { 0.25 * val })
            }
        }
    }

    /// Compute the derivative of the activation function
    pub fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Identity => Array2::ones(x.dim()),
            Activation::Logistic => {
                let activated = self.apply(x);
                &activated * &(1.0 - &activated)
            }
            Activation::Tanh => {
                let activated = self.apply(x);
                1.0 - &activated * &activated
            }
            Activation::Relu => x.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 }),
            Activation::Elu => x.mapv(|val| if val > 0.0 { 1.0 } else { 1.0 * val.exp() }),
            Activation::Swish => x.mapv(|val| {
                let sigmoid = 1.0 / (1.0 + (-val).exp());
                sigmoid * (1.0 + val * (1.0 - sigmoid))
            }),
            Activation::Gelu => x.mapv(|val| {
                let norm_cdf = 0.5 * (1.0 + erf_approx(val / SQRT_2));
                let norm_pdf = (1.0 / (2.0 * PI).sqrt()) * (-val * val / 2.0).exp();
                norm_cdf + val * norm_pdf
            }),
            Activation::Mish => x.mapv(|val| {
                let sp = softplus(val);
                let tanh_sp = sp.tanh();
                let sigmoid = 1.0 / (1.0 + (-val).exp());
                sigmoid * (1.0 + val * (1.0 - tanh_sp * tanh_sp))
            }),
            Activation::LeakyRelu => x.mapv(|val| if val > 0.0 { 1.0 } else { 0.01 }),
            Activation::PRelu => {
                // Note: PReLU derivatives depend on learned parameters
                // This is a fallback with alpha=0.25
                x.mapv(|val| if val > 0.0 { 1.0 } else { 0.25 })
            }
        }
    }

    /// Apply the activation function to a scalar value (for individual elements)
    pub fn forward(&self, x: f64) -> f64 {
        match self {
            Activation::Identity => x,
            Activation::Logistic => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
            Activation::Elu => {
                if x > 0.0 {
                    x
                } else {
                    1.0 * (x.exp() - 1.0)
                }
            }
            Activation::Swish => x * (1.0 / (1.0 + (-x).exp())),
            Activation::Gelu => 0.5 * x * (1.0 + erf_approx(x / SQRT_2)),
            Activation::Mish => x * softplus(x).tanh(),
            Activation::LeakyRelu => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::PRelu => {
                // Note: PReLU should be used as a separate parameterized layer
                // This is a fallback with alpha=0.25 (similar to default initialization)
                if x > 0.0 {
                    x
                } else {
                    0.25 * x
                }
            }
        }
    }

    /// Compute the derivative of the activation function for 1D arrays
    pub fn derivative_1d(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::Identity => Array1::ones(x.len()),
            Activation::Logistic => {
                let activated = self.apply_1d(x);
                &activated * &(1.0 - &activated)
            }
            Activation::Tanh => {
                let activated = self.apply_1d(x);
                1.0 - &activated * &activated
            }
            Activation::Relu => x.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 }),
            Activation::Elu => x.mapv(|val| if val > 0.0 { 1.0 } else { 1.0 * val.exp() }),
            Activation::Swish => x.mapv(|val| {
                let sigmoid = 1.0 / (1.0 + (-val).exp());
                sigmoid * (1.0 + val * (1.0 - sigmoid))
            }),
            Activation::Gelu => x.mapv(|val| {
                let norm_cdf = 0.5 * (1.0 + erf_approx(val / SQRT_2));
                let norm_pdf = (1.0 / (2.0 * PI).sqrt()) * (-val * val / 2.0).exp();
                norm_cdf + val * norm_pdf
            }),
            Activation::Mish => x.mapv(|val| {
                let sp = softplus(val);
                let tanh_sp = sp.tanh();
                let sigmoid = 1.0 / (1.0 + (-val).exp());
                sigmoid * (1.0 + val * (1.0 - tanh_sp * tanh_sp))
            }),
            Activation::LeakyRelu => x.mapv(|val| if val > 0.0 { 1.0 } else { 0.01 }),
            Activation::PRelu => {
                // Note: PReLU derivatives depend on learned parameters
                // This is a fallback with alpha=0.25
                x.mapv(|val| if val > 0.0 { 1.0 } else { 0.25 })
            }
        }
    }
}

/// Helper function for error function approximation (used in GELU)
/// Uses Abramowitz and Stegun approximation: erf(x) ≈ 1 - (1 + a1*x + a2*x² + a3*x³ + a4*x⁴)^(-4)
fn erf_approx(x: f64) -> f64 {
    const A1: f64 = 0.278393;
    const A2: f64 = 0.230389;
    const A3: f64 = 0.000972;
    const A4: f64 = 0.078108;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 + A1 * x + A2 * x * x + A3 * x * x * x + A4 * x * x * x * x;
    let result = 1.0 - 1.0 / (t * t * t * t);

    sign * result
}

/// Softplus function: log(1 + exp(x))
/// Used in Mish activation and other contexts
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        // For large values, exp(x) dominates and log(1 + exp(x)) ≈ x
        x
    } else if x < -20.0 {
        // For very negative values, exp(x) is near 0 and log(1 + exp(x)) ≈ exp(x)
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Softmax activation function for multi-class classification
pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(x.dim());

    Zip::from(x.axis_iter(scirs2_core::ndarray::Axis(0)))
        .and(result.axis_iter_mut(scirs2_core::ndarray::Axis(0)))
        .for_each(|input_row, mut output_row| {
            // Subtract max for numerical stability
            let max_val = input_row.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let exp_vals: Array1<f64> = input_row.mapv(|val| (val - max_val).exp());
            let sum_exp = exp_vals.sum();

            Zip::from(&exp_vals)
                .and(&mut output_row)
                .for_each(|&exp_val, output| {
                    *output = exp_val / sum_exp;
                });
        });

    result
}

/// Softmax activation function for 1D arrays
pub fn softmax_1d(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let exp_vals: Array1<f64> = x.mapv(|val| (val - max_val).exp());
    let sum_exp = exp_vals.sum();
    exp_vals / sum_exp
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array1, Array2};

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
    fn test_identity_activation() {
        let x = array![[1.0, -2.0, 3.0], [0.5, -1.5, 2.5]];
        let result = Activation::Identity.apply(&x);
        assert_arrays_close(&result, &x, 1e-10);
    }

    #[test]
    fn test_logistic_activation() {
        let x = array![[0.0], [1.0], [-1.0]];
        let result = Activation::Logistic.apply(&x);

        assert_abs_diff_eq!(result[[0, 0]], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(
            result[[1, 0]],
            1.0 / (1.0 + std::f64::consts::E.powf(-1.0)),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            result[[2, 0]],
            1.0 / (1.0 + std::f64::consts::E.powf(1.0)),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_tanh_activation() {
        let x = array![[0.0], [1.0], [-1.0]];
        let result = Activation::Tanh.apply(&x);

        assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 1.0_f64.tanh(), epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 0]], (-1.0_f64).tanh(), epsilon = 1e-10);
    }

    #[test]
    fn test_relu_activation() {
        let x = array![[-1.0, 0.0, 1.0, 2.0]];
        let result = Activation::Relu.apply(&x);
        let expected = array![[0.0, 0.0, 1.0, 2.0]];

        assert_arrays_close(&result, &expected, 1e-10);
    }

    #[test]
    fn test_logistic_derivative() {
        let x = array![[0.0], [1.0]];
        let derivative = Activation::Logistic.derivative(&x);

        // For logistic: f'(x) = f(x) * (1 - f(x))
        let activated = Activation::Logistic.apply(&x);
        let expected = &activated * &(1.0 - &activated);

        assert_arrays_close(&derivative, &expected, 1e-10);
    }

    #[test]
    fn test_relu_derivative() {
        let x = array![[-1.0, 0.0, 1.0, 2.0]];
        let derivative = Activation::Relu.derivative(&x);
        let expected = array![[0.0, 0.0, 1.0, 1.0]];

        assert_arrays_close(&derivative, &expected, 1e-10);
    }

    #[test]
    fn test_softmax_function() {
        let x = array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]];
        let result = softmax(&x);

        // Check that each row sums to 1
        for row in result.axis_iter(scirs2_core::ndarray::Axis(0)) {
            assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-10);
        }

        // Check that all values are positive
        for &val in result.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_softmax_1d() {
        let x = array![1.0, 2.0, 3.0];
        let result = softmax_1d(&x);

        // Check that it sums to 1
        assert_abs_diff_eq!(result.sum(), 1.0, epsilon = 1e-10);

        // Check that all values are positive
        for &val in result.iter() {
            assert!(val >= 0.0);
        }

        // Check that higher inputs give higher outputs
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_elu_activation() {
        let x = array![[-2.0, -1.0, 0.0, 1.0, 2.0]];
        let result = Activation::Elu.apply(&x);

        // For positive values, ELU should be identity
        assert_abs_diff_eq!(result[[0, 3]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 4]], 2.0, epsilon = 1e-10);

        // For negative values, should be alpha * (exp(x) - 1)
        let expected_neg2 = 1.0 * ((-2.0_f64).exp() - 1.0);
        assert_abs_diff_eq!(result[[0, 0]], expected_neg2, epsilon = 1e-10);

        // At zero, should be continuous
        assert_abs_diff_eq!(result[[0, 2]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_swish_activation() {
        let x = array![[0.0, 1.0, -1.0]];
        let result = Activation::Swish.apply(&x);

        // Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-10);

        // Swish(x) = x * sigmoid(x), check properties
        for i in 0..x.ncols() {
            let val = x[[0, i]];
            let sigmoid = 1.0 / (1.0 + (-val).exp());
            let expected = val * sigmoid;
            assert_abs_diff_eq!(result[[0, i]], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gelu_activation() {
        let x = array![[0.0, 1.0, -1.0]];
        let result = Activation::Gelu.apply(&x);

        // GELU(0) = 0
        assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-6);

        // GELU should be approximately 0.841 * x for x = 1
        assert!(result[[0, 1]] > 0.8);
        assert!(result[[0, 1]] < 0.9);

        // GELU should be negative for negative inputs but closer to 0
        assert!(result[[0, 2]] < 0.0);
        assert!(result[[0, 2]] > -0.2);
    }

    #[test]
    fn test_mish_activation() {
        let x = array![[0.0, 1.0, -1.0, 2.0]];
        let result = Activation::Mish.apply(&x);

        // Mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) ≈ 0
        assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-6);

        // For positive values, Mish should be close to the input but slightly different
        assert!(result[[0, 1]] > 0.8);
        assert!(result[[0, 1]] < 1.0);

        // For large positive values, should approach the input
        assert!(result[[0, 3]] > 1.9);
        assert!(result[[0, 3]] < 2.0);
    }

    #[test]
    fn test_leaky_relu_activation() {
        let x = array![[-2.0, -1.0, 0.0, 1.0, 2.0]];
        let result = Activation::LeakyRelu.apply(&x);

        // For positive values, should be identity
        assert_abs_diff_eq!(result[[0, 3]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 4]], 2.0, epsilon = 1e-10);

        // For negative values, should be 0.01 * x
        assert_abs_diff_eq!(result[[0, 0]], -0.02, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], -0.01, epsilon = 1e-10);

        // At zero
        assert_abs_diff_eq!(result[[0, 2]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_prelu_activation() {
        let x = array![[-2.0, -1.0, 0.0, 1.0, 2.0]];
        let result = Activation::PRelu.apply(&x);

        // For positive values, should be identity
        assert_abs_diff_eq!(result[[0, 3]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 4]], 2.0, epsilon = 1e-10);

        // For negative values, should be 0.25 * x (fallback alpha)
        assert_abs_diff_eq!(result[[0, 0]], -0.5, epsilon = 1e-10); // -2.0 * 0.25
        assert_abs_diff_eq!(result[[0, 1]], -0.25, epsilon = 1e-10); // -1.0 * 0.25

        // At zero
        assert_abs_diff_eq!(result[[0, 2]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_activation_derivatives() {
        let x = array![[0.5, -0.5, 1.0, -1.0]];

        // Test that derivatives are computed correctly for new activations
        let elu_deriv = Activation::Elu.derivative(&x);
        assert_abs_diff_eq!(elu_deriv[[0, 0]], 1.0, epsilon = 1e-10); // positive: 1
        assert_abs_diff_eq!(elu_deriv[[0, 2]], 1.0, epsilon = 1e-10); // positive: 1

        let leaky_deriv = Activation::LeakyRelu.derivative(&x);
        assert_abs_diff_eq!(leaky_deriv[[0, 0]], 1.0, epsilon = 1e-10); // positive: 1
        assert_abs_diff_eq!(leaky_deriv[[0, 1]], 0.01, epsilon = 1e-10); // negative: 0.01
    }

    #[test]
    fn test_helper_functions() {
        // Test erf approximation
        assert_abs_diff_eq!(erf_approx(0.0), 0.0, epsilon = 1e-3);
        assert!(erf_approx(1.0) > 0.8 && erf_approx(1.0) < 0.9);
        assert!(erf_approx(-1.0) < -0.8 && erf_approx(-1.0) > -0.9);

        // Test softplus
        assert_abs_diff_eq!(softplus(0.0), 2.0_f64.ln(), epsilon = 1e-10);
        assert!(softplus(100.0) - 100.0 < 1e-10); // Should be approximately x for large x
        assert!(softplus(-100.0) < 1e-40); // Should be approximately 0 for very negative x
    }
}
