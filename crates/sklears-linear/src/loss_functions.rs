//! Pluggable Loss Functions for Linear Models
//!
//! This module implements various loss functions that can be used with the modular framework.
//! All loss functions implement the LossFunction trait for consistency and pluggability.

use crate::modular_framework::LossFunction;
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Mean Squared Error loss for regression
#[derive(Debug, Clone)]
pub struct SquaredLoss;

impl LossFunction for SquaredLoss {
    fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let diff = y_pred - y_true;
        let mse = diff.mapv(|x| x * x).sum() / (2.0 * y_true.len() as Float);
        Ok(mse)
    }

    fn loss_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        Ok((y_pred - y_true) / (y_true.len() as Float))
    }

    fn name(&self) -> &'static str {
        "SquaredLoss"
    }
}

/// Mean Absolute Error loss for robust regression
#[derive(Debug, Clone)]
pub struct AbsoluteLoss;

impl LossFunction for AbsoluteLoss {
    fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let mae = (y_pred - y_true).mapv(|x| x.abs()).sum() / (y_true.len() as Float);
        Ok(mae)
    }

    fn loss_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let derivative = (y_pred - y_true).mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        });
        Ok(derivative / (y_true.len() as Float))
    }

    fn name(&self) -> &'static str {
        "AbsoluteLoss"
    }
}

/// Huber loss for robust regression (combination of squared and absolute loss)
#[derive(Debug, Clone)]
pub struct HuberLoss {
    /// Threshold parameter controlling the transition between squared and absolute loss
    pub delta: Float,
}

impl HuberLoss {
    /// Create a new Huber loss with the specified delta parameter
    pub fn new(delta: Float) -> Self {
        Self { delta }
    }

    /// Create a Huber loss with default delta = 1.0
    pub fn default() -> Self {
        Self::new(1.0)
    }
}

impl LossFunction for HuberLoss {
    fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let residuals = y_pred - y_true;
        let loss_sum = residuals
            .mapv(|r| {
                let abs_r = r.abs();
                if abs_r <= self.delta {
                    0.5 * r * r
                } else {
                    self.delta * abs_r - 0.5 * self.delta * self.delta
                }
            })
            .sum();

        Ok(loss_sum / (y_true.len() as Float))
    }

    fn loss_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let residuals = y_pred - y_true;
        let derivative = residuals.mapv(|r| {
            if r.abs() <= self.delta {
                r
            } else if r > 0.0 {
                self.delta
            } else {
                -self.delta
            }
        });

        Ok(derivative / (y_true.len() as Float))
    }

    fn name(&self) -> &'static str {
        "HuberLoss"
    }
}

/// Logistic loss for binary classification
#[derive(Debug, Clone)]
pub struct LogisticLoss;

impl LossFunction for LogisticLoss {
    fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        // y_pred are logits, y_true should be in {-1, 1} or {0, 1}
        let loss_sum = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &pred)| {
                // Convert y_true to {-1, 1} format if it's in {0, 1}
                let y_adjusted = if y == 0.0 { -1.0 } else { y };

                // Numerically stable computation: log(1 + exp(-y * pred))
                let margin = y_adjusted * pred;
                if margin > 0.0 {
                    (1.0 + (-margin).exp()).ln()
                } else {
                    -margin + (1.0 + margin.exp()).ln()
                }
            })
            .sum::<Float>();

        Ok(loss_sum / (y_true.len() as Float))
    }

    fn loss_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let derivative = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &pred)| {
                // Convert y_true to {-1, 1} format if it's in {0, 1}
                let y_adjusted = if y == 0.0 { -1.0 } else { y };

                // Derivative: -y / (1 + exp(y * pred))
                let margin = y_adjusted * pred;
                -y_adjusted / (1.0 + margin.exp())
            })
            .collect::<Vec<Float>>();

        let result = Array1::from_vec(derivative) / (y_true.len() as Float);
        Ok(result)
    }

    fn is_classification(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "LogisticLoss"
    }
}

/// Hinge loss for Support Vector Machines
#[derive(Debug, Clone)]
pub struct HingeLoss;

impl LossFunction for HingeLoss {
    fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        // y_true should be in {-1, 1}, y_pred are decision function values
        let loss_sum = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &pred)| {
                let y_adjusted = if y == 0.0 { -1.0 } else { y };
                let margin = y_adjusted * pred;
                (1.0 - margin).max(0.0)
            })
            .sum::<Float>();

        Ok(loss_sum / (y_true.len() as Float))
    }

    fn loss_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let derivative = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &pred)| {
                let y_adjusted = if y == 0.0 { -1.0 } else { y };
                let margin = y_adjusted * pred;
                if margin < 1.0 {
                    -y_adjusted
                } else {
                    0.0
                }
            })
            .collect::<Vec<Float>>();

        let result = Array1::from_vec(derivative) / (y_true.len() as Float);
        Ok(result)
    }

    fn is_classification(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "HingeLoss"
    }
}

/// Squared Hinge loss (smooth variant of hinge loss)
#[derive(Debug, Clone)]
pub struct SquaredHingeLoss;

impl LossFunction for SquaredHingeLoss {
    fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let loss_sum = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &pred)| {
                let y_adjusted = if y == 0.0 { -1.0 } else { y };
                let margin = y_adjusted * pred;
                let hinge = (1.0 - margin).max(0.0);
                hinge * hinge
            })
            .sum::<Float>();

        Ok(loss_sum / (y_true.len() as Float))
    }

    fn loss_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let derivative = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &pred)| {
                let y_adjusted = if y == 0.0 { -1.0 } else { y };
                let margin = y_adjusted * pred;
                if margin < 1.0 {
                    -2.0 * y_adjusted * (1.0 - margin)
                } else {
                    0.0
                }
            })
            .collect::<Vec<Float>>();

        let result = Array1::from_vec(derivative) / (y_true.len() as Float);
        Ok(result)
    }

    fn is_classification(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "SquaredHingeLoss"
    }
}

/// Quantile loss for quantile regression
#[derive(Debug, Clone)]
pub struct QuantileLoss {
    /// Quantile parameter (between 0 and 1)
    pub quantile: Float,
}

impl QuantileLoss {
    /// Create a new quantile loss with the specified quantile
    pub fn new(quantile: Float) -> Result<Self> {
        if quantile <= 0.0 || quantile >= 1.0 {
            return Err(SklearsError::InvalidParameter {
                name: "quantile".to_string(),
                reason: format!("Quantile must be between 0 and 1, got {}", quantile),
            });
        }
        Ok(Self { quantile })
    }

    /// Create median regression (quantile = 0.5)
    pub fn median() -> Self {
        Self { quantile: 0.5 }
    }
}

impl LossFunction for QuantileLoss {
    fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let residuals = y_true - y_pred;
        let loss_sum = residuals
            .mapv(|r| {
                if r >= 0.0 {
                    self.quantile * r
                } else {
                    (self.quantile - 1.0) * r
                }
            })
            .sum();

        Ok(loss_sum / (y_true.len() as Float))
    }

    fn loss_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let residuals = y_true - y_pred;
        let derivative = residuals.mapv(|r| {
            if r > 0.0 {
                -self.quantile
            } else if r < 0.0 {
                -(self.quantile - 1.0)
            } else {
                0.0 // Subgradient at 0
            }
        });

        Ok(derivative / (y_true.len() as Float))
    }

    fn name(&self) -> &'static str {
        "QuantileLoss"
    }
}

/// Epsilon-insensitive loss for Support Vector Regression
#[derive(Debug, Clone)]
pub struct EpsilonInsensitiveLoss {
    /// Epsilon parameter (tolerance for errors)
    pub epsilon: Float,
}

impl EpsilonInsensitiveLoss {
    /// Create a new epsilon-insensitive loss
    pub fn new(epsilon: Float) -> Result<Self> {
        if epsilon < 0.0 {
            return Err(SklearsError::InvalidParameter {
                name: "epsilon".to_string(),
                reason: format!("Epsilon must be non-negative, got {}", epsilon),
            });
        }
        Ok(Self { epsilon })
    }
}

impl LossFunction for EpsilonInsensitiveLoss {
    fn loss(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let loss_sum = (y_true - y_pred)
            .mapv(|r| (r.abs() - self.epsilon).max(0.0))
            .sum();

        Ok(loss_sum / (y_true.len() as Float))
    }

    fn loss_derivative(
        &self,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }

        let residuals = y_true - y_pred;
        let derivative = residuals.mapv(|r| {
            if r > self.epsilon {
                -1.0
            } else if r < -self.epsilon {
                1.0
            } else {
                0.0
            }
        });

        Ok(derivative / (y_true.len() as Float))
    }

    fn name(&self) -> &'static str {
        "EpsilonInsensitiveLoss"
    }
}

/// Factory for creating common loss functions
pub struct LossFactory;

impl LossFactory {
    /// Create a squared loss (MSE)
    pub fn squared() -> Box<dyn LossFunction> {
        Box::new(SquaredLoss)
    }

    /// Create an absolute loss (MAE)
    pub fn absolute() -> Box<dyn LossFunction> {
        Box::new(AbsoluteLoss)
    }

    /// Create a Huber loss with specified delta
    pub fn huber(delta: Float) -> Box<dyn LossFunction> {
        Box::new(HuberLoss::new(delta))
    }

    /// Create a logistic loss for binary classification
    pub fn logistic() -> Box<dyn LossFunction> {
        Box::new(LogisticLoss)
    }

    /// Create a hinge loss for SVM
    pub fn hinge() -> Box<dyn LossFunction> {
        Box::new(HingeLoss)
    }

    /// Create a squared hinge loss
    pub fn squared_hinge() -> Box<dyn LossFunction> {
        Box::new(SquaredHingeLoss)
    }

    /// Create a quantile loss
    pub fn quantile(quantile: Float) -> Result<Box<dyn LossFunction>> {
        Ok(Box::new(QuantileLoss::new(quantile)?))
    }

    /// Create an epsilon-insensitive loss for SVR
    pub fn epsilon_insensitive(epsilon: Float) -> Result<Box<dyn LossFunction>> {
        Ok(Box::new(EpsilonInsensitiveLoss::new(epsilon)?))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_squared_loss() {
        let loss = SquaredLoss;
        let y_true = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = Array::from_vec(vec![1.1, 1.9, 3.1]);

        let loss_value = loss.loss(&y_true, &y_pred).unwrap();
        let expected = ((0.1 * 0.1) + (0.1 * 0.1) + (0.1 * 0.1)) / (2.0 * 3.0);
        assert!((loss_value - expected).abs() < 1e-10);

        let derivative = loss.loss_derivative(&y_true, &y_pred).unwrap();
        let expected_grad = Array::from_vec(vec![0.1, -0.1, 0.1]) / 3.0;
        for (actual, expected) in derivative.iter().zip(expected_grad.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_absolute_loss() {
        let loss = AbsoluteLoss;
        let y_true = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = Array::from_vec(vec![1.2, 1.8, 3.1]);

        let loss_value = loss.loss(&y_true, &y_pred).unwrap();
        let expected = (0.2 + 0.2 + 0.1) / 3.0;
        assert!((loss_value - expected).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss() {
        let loss = HuberLoss::new(1.0);
        let y_true = Array::from_vec(vec![0.0, 0.0]);
        let y_pred = Array::from_vec(vec![0.5, 2.0]); // First within delta, second outside

        let loss_value = loss.loss(&y_true, &y_pred).unwrap();
        // First: 0.5 * 0.5^2 = 0.125
        // Second: 1.0 * 2.0 - 0.5 * 1.0^2 = 1.5
        let expected = (0.125 + 1.5) / 2.0;
        assert!((loss_value - expected).abs() < 1e-10);
    }

    #[test]
    fn test_logistic_loss() {
        let loss = LogisticLoss;
        let y_true = Array::from_vec(vec![1.0, -1.0]);
        let y_pred = Array::from_vec(vec![2.0, -2.0]); // Strong correct predictions

        let loss_value = loss.loss(&y_true, &y_pred).unwrap();
        // Should be small for correct predictions
        assert!(loss_value < 0.5);
    }

    #[test]
    fn test_quantile_loss() {
        let loss = QuantileLoss::new(0.7).unwrap();
        let y_true = Array::from_vec(vec![1.0, 2.0]);
        let y_pred = Array::from_vec(vec![0.5, 2.5]); // Under-predict, over-predict

        let loss_value = loss.loss(&y_true, &y_pred).unwrap();
        // Under-prediction: 0.7 * 0.5 = 0.35
        // Over-prediction: (0.7 - 1.0) * (-0.5) = 0.15
        let expected = (0.35 + 0.15) / 2.0;
        assert!((loss_value - expected).abs() < 1e-10);
    }

    #[test]
    fn test_loss_factory() {
        let squared = LossFactory::squared();
        assert_eq!(squared.name(), "SquaredLoss");

        let huber = LossFactory::huber(1.5);
        assert_eq!(huber.name(), "HuberLoss");

        let quantile = LossFactory::quantile(0.8).unwrap();
        assert_eq!(quantile.name(), "QuantileLoss");
    }

    #[test]
    fn test_dimension_mismatch() {
        let loss = SquaredLoss;
        let y_true = Array::from_vec(vec![1.0, 2.0]);
        let y_pred = Array::from_vec(vec![1.0, 2.0, 3.0]);

        let result = loss.loss(&y_true, &y_pred);
        assert!(result.is_err());
    }
}
