//! Custom loss function framework for tree-based algorithms
//!
//! This module provides a flexible framework for defining custom loss functions
//! that can be used with gradient boosting and other tree-based algorithms.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, types::Float};
use std::fmt::Debug;

/// Trait for loss functions used in tree-based algorithms
pub trait LossFunction: Debug + Clone + Send + Sync {
    /// Compute the loss given true and predicted values
    fn loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>;

    /// Compute the gradient (negative gradient for gradient boosting)
    fn gradient(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>>;

    /// Compute the hessian (second derivative)
    fn hessian(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>>;

    /// Initialize predictions (e.g., for gradient boosting)
    fn init_prediction(&self, y_true: &Array1<f64>) -> Result<f64>;

    /// Check if the loss function is suitable for classification
    fn is_classification(&self) -> bool;

    /// Check if the loss function is suitable for regression
    fn is_regression(&self) -> bool;

    /// Get the name of the loss function
    fn name(&self) -> &'static str;
}

/// Squared loss for regression
#[derive(Debug, Clone)]
pub struct SquaredLoss;

impl LossFunction for SquaredLoss {
    fn loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let diff = y_true - y_pred;
        Ok(0.5 * diff.mapv(|x| x * x).sum() / y_true.len() as f64)
    }

    fn gradient(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        Ok(y_pred - y_true) // Negative gradient for boosting
    }

    fn hessian(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        Ok(Array1::ones(y_true.len()))
    }

    fn init_prediction(&self, y_true: &Array1<f64>) -> Result<f64> {
        Ok(y_true.mean().unwrap_or(0.0))
    }

    fn is_classification(&self) -> bool {
        false
    }

    fn is_regression(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "squared_loss"
    }
}

/// Absolute loss (L1 loss) for regression
#[derive(Debug, Clone)]
pub struct AbsoluteLoss;

impl LossFunction for AbsoluteLoss {
    fn loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let diff = y_true - y_pred;
        Ok(diff.mapv(|x| x.abs()).sum() / y_true.len() as f64)
    }

    fn gradient(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let diff = y_pred - y_true;
        Ok(diff.mapv(|x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }))
    }

    fn hessian(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        // Hessian of absolute loss is zero everywhere except at the point of non-differentiability
        // We use a small constant to avoid numerical issues
        Ok(Array1::from_elem(y_true.len(), 1e-6))
    }

    fn init_prediction(&self, y_true: &Array1<f64>) -> Result<f64> {
        // Use median for absolute loss
        let mut sorted = y_true.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        Ok(median)
    }

    fn is_classification(&self) -> bool {
        false
    }

    fn is_regression(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "absolute_loss"
    }
}

/// Huber loss for robust regression
#[derive(Debug, Clone)]
pub struct HuberLoss {
    pub delta: f64,
}

impl HuberLoss {
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl LossFunction for HuberLoss {
    fn loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let diff = y_true - y_pred;
        let loss = diff.mapv(|x| {
            let abs_x = x.abs();
            if abs_x <= self.delta {
                0.5 * x * x
            } else {
                self.delta * abs_x - 0.5 * self.delta * self.delta
            }
        });
        Ok(loss.sum() / y_true.len() as f64)
    }

    fn gradient(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let diff = y_pred - y_true;
        Ok(diff.mapv(|x| {
            if x.abs() <= self.delta {
                x
            } else if x > 0.0 {
                self.delta
            } else {
                -self.delta
            }
        }))
    }

    fn hessian(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let diff = y_pred - y_true;
        Ok(diff.mapv(|x| if x.abs() <= self.delta { 1.0 } else { 0.0 }))
    }

    fn init_prediction(&self, y_true: &Array1<f64>) -> Result<f64> {
        Ok(y_true.mean().unwrap_or(0.0))
    }

    fn is_classification(&self) -> bool {
        false
    }

    fn is_regression(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "huber_loss"
    }
}

/// Quantile loss for quantile regression
#[derive(Debug, Clone)]
pub struct QuantileLoss {
    pub alpha: f64, // The quantile to predict (0 < alpha < 1)
}

impl QuantileLoss {
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "alpha must be between 0 and 1".to_string(),
            ));
        }
        Ok(Self { alpha })
    }
}

impl Default for QuantileLoss {
    fn default() -> Self {
        Self { alpha: 0.5 } // Median
    }
}

impl LossFunction for QuantileLoss {
    fn loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let diff = y_true - y_pred;
        let loss = diff.mapv(|x| {
            if x >= 0.0 {
                self.alpha * x
            } else {
                (self.alpha - 1.0) * x
            }
        });
        Ok(loss.sum() / y_true.len() as f64)
    }

    fn gradient(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let diff = y_pred - y_true;
        Ok(diff.mapv(|x| {
            if x > 0.0 {
                1.0 - self.alpha
            } else if x < 0.0 {
                -self.alpha
            } else {
                0.0
            }
        }))
    }

    fn hessian(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        // Hessian is zero for quantile loss
        Ok(Array1::from_elem(y_true.len(), 1e-6))
    }

    fn init_prediction(&self, y_true: &Array1<f64>) -> Result<f64> {
        // Use the alpha-quantile as initial prediction
        let mut sorted = y_true.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let index = (self.alpha * (sorted.len() - 1) as f64).round() as usize;
        Ok(sorted[index.min(sorted.len() - 1)])
    }

    fn is_classification(&self) -> bool {
        false
    }

    fn is_regression(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "quantile_loss"
    }
}

/// Logistic loss for binary classification
#[derive(Debug, Clone)]
pub struct LogisticLoss;

impl LossFunction for LogisticLoss {
    fn loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let loss = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let y_t = if y_t > 0.5 { 1.0 } else { -1.0 }; // Convert to {-1, 1}
                (1.0 + (-y_t * y_p).exp()).ln()
            })
            .sum::<f64>();
        
        Ok(loss / y_true.len() as f64)
    }

    fn gradient(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let grad = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let y_t = if y_t > 0.5 { 1.0 } else { -1.0 }; // Convert to {-1, 1}
                -y_t / (1.0 + (y_t * y_p).exp())
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from(grad))
    }

    fn hessian(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let hess = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let y_t = if y_t > 0.5 { 1.0 } else { -1.0 }; // Convert to {-1, 1}
                let exp_val = (y_t * y_p).exp();
                exp_val / (1.0 + exp_val).powi(2)
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from(hess))
    }

    fn init_prediction(&self, y_true: &Array1<f64>) -> Result<f64> {
        let positive_rate = y_true.iter().filter(|&&y| y > 0.5).count() as f64 / y_true.len() as f64;
        Ok((positive_rate / (1.0 - positive_rate)).ln())
    }

    fn is_classification(&self) -> bool {
        true
    }

    fn is_regression(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "logistic_loss"
    }
}

/// Exponential loss (used in AdaBoost)
#[derive(Debug, Clone)]
pub struct ExponentialLoss;

impl LossFunction for ExponentialLoss {
    fn loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let loss = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let y_t = if y_t > 0.5 { 1.0 } else { -1.0 }; // Convert to {-1, 1}
                (-y_t * y_p).exp()
            })
            .sum::<f64>();

        Ok(loss / y_true.len() as f64)
    }

    fn gradient(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let grad = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let y_t = if y_t > 0.5 { 1.0 } else { -1.0 }; // Convert to {-1, 1}
                -y_t * (-y_t * y_p).exp()
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from(grad))
    }

    fn hessian(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let hess = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let y_t = if y_t > 0.5 { 1.0 } else { -1.0 }; // Convert to {-1, 1}
                (-y_t * y_p).exp()
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from(hess))
    }

    fn init_prediction(&self, y_true: &Array1<f64>) -> Result<f64> {
        Ok(0.0) // Start with neutral prediction
    }

    fn is_classification(&self) -> bool {
        true
    }

    fn is_regression(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "exponential_loss"
    }
}

/// Focal loss for imbalanced classification
#[derive(Debug, Clone)]
pub struct FocalLoss {
    pub alpha: f64, // Weighting factor for rare class
    pub gamma: f64, // Focusing parameter
}

impl FocalLoss {
    pub fn new(alpha: f64, gamma: f64) -> Self {
        Self { alpha, gamma }
    }
}

impl Default for FocalLoss {
    fn default() -> Self {
        Self { alpha: 0.25, gamma: 2.0 }
    }
}

impl LossFunction for FocalLoss {
    fn loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let loss = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let y_t = if y_t > 0.5 { 1.0 } else { 0.0 };
                let p = 1.0 / (1.0 + (-y_p).exp()); // Sigmoid
                let ce = if y_t == 1.0 {
                    -(p.max(1e-15)).ln()
                } else {
                    -((1.0 - p).max(1e-15)).ln()
                };
                let pt = if y_t == 1.0 { p } else { 1.0 - p };
                let alpha_t = if y_t == 1.0 { self.alpha } else { 1.0 - self.alpha };
                alpha_t * (1.0 - pt).powf(self.gamma) * ce
            })
            .sum::<f64>();

        Ok(loss / y_true.len() as f64)
    }

    fn gradient(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let grad = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let y_t = if y_t > 0.5 { 1.0 } else { 0.0 };
                let p = 1.0 / (1.0 + (-y_p).exp()); // Sigmoid
                let pt = if y_t == 1.0 { p } else { 1.0 - p };
                let alpha_t = if y_t == 1.0 { self.alpha } else { 1.0 - self.alpha };
                
                // Simplified gradient (approximation)
                let factor = alpha_t * (1.0 - pt).powf(self.gamma);
                if y_t == 1.0 {
                    -factor * (1.0 - p)
                } else {
                    factor * p
                }
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from(grad))
    }

    fn hessian(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        if y_true.len() != y_pred.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        // Simplified hessian (approximation)
        let hess = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let p = 1.0 / (1.0 + (-y_p).exp()); // Sigmoid
                p * (1.0 - p) // Standard logistic hessian
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from(hess))
    }

    fn init_prediction(&self, y_true: &Array1<f64>) -> Result<f64> {
        let positive_rate = y_true.iter().filter(|&&y| y > 0.5).count() as f64 / y_true.len() as f64;
        Ok((positive_rate / (1.0 - positive_rate)).ln())
    }

    fn is_classification(&self) -> bool {
        true
    }

    fn is_regression(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "focal_loss"
    }
}

/// Custom loss function implementation
pub struct CustomLoss<F, G, H, I>
where
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync,
    G: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
    H: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
    I: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
{
    loss_fn: F,
    gradient_fn: G,
    hessian_fn: H,
    init_fn: I,
    name: &'static str,
    is_classification: bool,
}

impl<F, G, H, I> CustomLoss<F, G, H, I>
where
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync,
    G: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
    H: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
    I: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
{
    pub fn new(
        loss_fn: F,
        gradient_fn: G,
        hessian_fn: H,
        init_fn: I,
        name: &'static str,
        is_classification: bool,
    ) -> Self {
        Self {
            loss_fn,
            gradient_fn,
            hessian_fn,
            init_fn,
            name,
            is_classification,
        }
    }
}

impl<F, G, H, I> Debug for CustomLoss<F, G, H, I>
where
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync,
    G: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
    H: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
    I: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomLoss")
            .field("name", &self.name)
            .field("is_classification", &self.is_classification)
            .finish()
    }
}

impl<F, G, H, I> Clone for CustomLoss<F, G, H, I>
where
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync + Clone,
    G: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync + Clone,
    H: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync + Clone,
    I: Fn(&Array1<f64>) -> Result<f64> + Send + Sync + Clone,
{
    fn clone(&self) -> Self {
        Self {
            loss_fn: self.loss_fn.clone(),
            gradient_fn: self.gradient_fn.clone(),
            hessian_fn: self.hessian_fn.clone(),
            init_fn: self.init_fn.clone(),
            name: self.name,
            is_classification: self.is_classification,
        }
    }
}

impl<F, G, H, I> LossFunction for CustomLoss<F, G, H, I>
where
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync,
    G: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
    H: Fn(&Array1<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
    I: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
{
    fn loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        (self.loss_fn)(y_true, y_pred)
    }

    fn gradient(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        (self.gradient_fn)(y_true, y_pred)
    }

    fn hessian(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Array1<f64>> {
        (self.hessian_fn)(y_true, y_pred)
    }

    fn init_prediction(&self, y_true: &Array1<f64>) -> Result<f64> {
        (self.init_fn)(y_true)
    }

    fn is_classification(&self) -> bool {
        self.is_classification
    }

    fn is_regression(&self) -> bool {
        !self.is_classification
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

/// Utility functions for loss functions
pub mod utils {
    use super::*;

    /// Create a squared loss function
    pub fn squared_loss() -> SquaredLoss {
        SquaredLoss
    }

    /// Create an absolute loss function
    pub fn absolute_loss() -> AbsoluteLoss {
        AbsoluteLoss
    }

    /// Create a Huber loss function with specified delta
    pub fn huber_loss(delta: f64) -> HuberLoss {
        HuberLoss::new(delta)
    }

    /// Create a quantile loss function for specified quantile
    pub fn quantile_loss(alpha: f64) -> Result<QuantileLoss> {
        QuantileLoss::new(alpha)
    }

    /// Create a logistic loss function
    pub fn logistic_loss() -> LogisticLoss {
        LogisticLoss
    }

    /// Create an exponential loss function
    pub fn exponential_loss() -> ExponentialLoss {
        ExponentialLoss
    }

    /// Create a focal loss function with specified parameters
    pub fn focal_loss(alpha: f64, gamma: f64) -> FocalLoss {
        FocalLoss::new(alpha, gamma)
    }

    /// Evaluate a loss function and return all derivatives
    pub fn evaluate_loss<L: LossFunction>(
        loss_fn: &L,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array1<f64>)> {
        let loss = loss_fn.loss(y_true, y_pred)?;
        let gradient = loss_fn.gradient(y_true, y_pred)?;
        let hessian = loss_fn.hessian(y_true, y_pred)?;
        Ok((loss, gradient, hessian))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_squared_loss() {
        let loss = SquaredLoss;
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.5, 2.5];

        let l = loss.loss(&y_true, &y_pred).unwrap();
        assert!((l - 0.25).abs() < 1e-10); // (0.25 + 0.25 + 0.25) / 3 = 0.25

        let grad = loss.gradient(&y_true, &y_pred).unwrap();
        assert_eq!(grad, array![0.5, 0.5, -0.5]);

        let hess = loss.hessian(&y_true, &y_pred).unwrap();
        assert_eq!(hess, array![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_absolute_loss() {
        let loss = AbsoluteLoss;
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.5, 2.5];

        let l = loss.loss(&y_true, &y_pred).unwrap();
        assert!((l - 0.5).abs() < 1e-10); // (0.5 + 0.5 + 0.5) / 3 = 0.5

        let grad = loss.gradient(&y_true, &y_pred).unwrap();
        assert_eq!(grad, array![1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_huber_loss() {
        let loss = HuberLoss::new(1.0);
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.5, 2.5];

        let l = loss.loss(&y_true, &y_pred).unwrap();
        assert!(l > 0.0);

        let grad = loss.gradient(&y_true, &y_pred).unwrap();
        assert_eq!(grad.len(), 3);
    }

    #[test]
    fn test_quantile_loss() {
        let loss = QuantileLoss::new(0.5).unwrap();
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.5, 2.5];

        let l = loss.loss(&y_true, &y_pred).unwrap();
        assert!(l > 0.0);

        let grad = loss.gradient(&y_true, &y_pred).unwrap();
        assert_eq!(grad.len(), 3);
    }

    #[test]
    fn test_logistic_loss() {
        let loss = LogisticLoss;
        let y_true = array![1.0, 0.0, 1.0];
        let y_pred = array![0.5, -0.5, 1.0];

        let l = loss.loss(&y_true, &y_pred).unwrap();
        assert!(l > 0.0);

        let grad = loss.gradient(&y_true, &y_pred).unwrap();
        assert_eq!(grad.len(), 3);
    }

    #[test]
    fn test_exponential_loss() {
        let loss = ExponentialLoss;
        let y_true = array![1.0, 0.0, 1.0];
        let y_pred = array![0.5, -0.5, 1.0];

        let l = loss.loss(&y_true, &y_pred).unwrap();
        assert!(l > 0.0);

        let grad = loss.gradient(&y_true, &y_pred).unwrap();
        assert_eq!(grad.len(), 3);
    }

    #[test]
    fn test_focal_loss() {
        let loss = FocalLoss::new(0.25, 2.0);
        let y_true = array![1.0, 0.0, 1.0];
        let y_pred = array![0.5, -0.5, 1.0];

        let l = loss.loss(&y_true, &y_pred).unwrap();
        assert!(l > 0.0);

        let grad = loss.gradient(&y_true, &y_pred).unwrap();
        assert_eq!(grad.len(), 3);
    }

    #[test]
    fn test_loss_properties() {
        let squared = SquaredLoss;
        assert!(squared.is_regression());
        assert!(!squared.is_classification());
        assert_eq!(squared.name(), "squared_loss");

        let logistic = LogisticLoss;
        assert!(!logistic.is_regression());
        assert!(logistic.is_classification());
        assert_eq!(logistic.name(), "logistic_loss");
    }

    #[test]
    fn test_utils() {
        let loss = utils::squared_loss();
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.5, 2.5];

        let (l, g, h) = utils::evaluate_loss(&loss, &y_true, &y_pred).unwrap();
        assert!(l > 0.0);
        assert_eq!(g.len(), 2);
        assert_eq!(h.len(), 2);
    }
}