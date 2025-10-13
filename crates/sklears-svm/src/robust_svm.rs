//! Robust Support Vector Machines with robust loss functions
//!
//! This module implements robust SVM variants that use loss functions designed
//! to be less sensitive to outliers and noisy data. The implemented loss functions
//! include Huber loss, ε-insensitive loss variants, and other robust formulations.

use crate::kernels::{Kernel, KernelType};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Robust loss function types
#[derive(Debug, Clone, PartialEq)]
pub enum RobustLoss {
    /// Huber loss - quadratic for small errors, linear for large errors
    /// delta parameter controls the threshold
    Huber { delta: Float },
    /// ε-insensitive loss with robustification
    EpsilonInsensitive { epsilon: Float },
    /// Squared Huber loss
    SquaredHuber { delta: Float },
    /// Smoothed Hinge loss for classification
    SmoothedHinge { gamma: Float },
    /// Pinball loss for quantile regression
    Pinball { tau: Float },
}

impl Default for RobustLoss {
    fn default() -> Self {
        RobustLoss::Huber { delta: 1.0 }
    }
}

/// Configuration for robust SVM
#[derive(Debug, Clone)]
pub struct RobustSVMConfig {
    /// Regularization parameter
    pub c: Float,
    /// Robust loss function
    pub loss: RobustLoss,
    /// Kernel function to use
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Random seed
    pub random_state: Option<u64>,
    /// Learning rate for gradient-based optimization
    pub learning_rate: Float,
    /// Learning rate decay
    pub learning_rate_decay: Float,
    /// Minimum learning rate
    pub min_learning_rate: Float,
}

impl Default for RobustSVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            loss: RobustLoss::default(),
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 1000,
            fit_intercept: true,
            random_state: None,
            learning_rate: 0.01,
            learning_rate_decay: 0.99,
            min_learning_rate: 1e-6,
        }
    }
}

/// Robust Support Vector Machine for classification and regression
#[derive(Debug)]
pub struct RobustSVM<State = Untrained> {
    config: RobustSVMConfig,
    state: PhantomData<State>,
    // Fitted attributes
    support_vectors_: Option<Array2<Float>>,
    alpha_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_in_: Option<usize>,
    n_iter_: Option<usize>,
}

impl RobustSVM<Untrained> {
    /// Create a new robust SVM
    pub fn new() -> Self {
        Self {
            config: RobustSVMConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            alpha_: None,
            intercept_: None,
            n_features_in_: None,
            n_iter_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the robust loss function
    pub fn loss(mut self, loss: RobustLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }
}

impl Default for RobustSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustLoss {
    /// Compute the loss value for given prediction error
    pub fn loss(&self, error: Float) -> Float {
        match self {
            RobustLoss::Huber { delta } => {
                if error.abs() <= *delta {
                    0.5 * error * error
                } else {
                    delta * (error.abs() - 0.5 * delta)
                }
            }
            RobustLoss::EpsilonInsensitive { epsilon } => (error.abs() - epsilon).max(0.0),
            RobustLoss::SquaredHuber { delta } => {
                if error.abs() <= *delta {
                    error * error
                } else {
                    2.0 * delta * error.abs() - delta * delta
                }
            }
            RobustLoss::SmoothedHinge { gamma } => {
                if error >= 1.0 {
                    0.0
                } else if error <= 1.0 - *gamma {
                    1.0 - error - 0.5 * gamma
                } else {
                    (1.0 - error) * (1.0 - error) / (2.0 * gamma)
                }
            }
            RobustLoss::Pinball { tau } => {
                if error >= 0.0 {
                    tau * error
                } else {
                    (tau - 1.0) * error
                }
            }
        }
    }

    /// Compute the derivative of the loss function
    pub fn derivative(&self, error: Float) -> Float {
        match self {
            RobustLoss::Huber { delta } => {
                if error.abs() <= *delta {
                    error
                } else {
                    delta * error.signum()
                }
            }
            RobustLoss::EpsilonInsensitive { epsilon } => {
                if error.abs() <= *epsilon {
                    0.0
                } else {
                    error.signum()
                }
            }
            RobustLoss::SquaredHuber { delta } => {
                if error.abs() <= *delta {
                    2.0 * error
                } else {
                    2.0 * delta * error.signum()
                }
            }
            RobustLoss::SmoothedHinge { gamma } => {
                if error >= 1.0 {
                    0.0
                } else if error <= 1.0 - *gamma {
                    -1.0
                } else {
                    -(1.0 - error) / gamma
                }
            }
            RobustLoss::Pinball { tau } => {
                if error >= 0.0 {
                    *tau
                } else {
                    tau - 1.0
                }
            }
        }
    }
}

impl Fit<Array2<Float>, Array1<Float>> for RobustSVM<Untrained> {
    type Fitted = RobustSVM<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Shape mismatch: X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize support vectors and coefficients
        let support_vectors = x.clone();
        let mut alpha = Array1::<Float>::zeros(n_samples);
        let mut intercept = 0.0;
        let mut current_lr = self.config.learning_rate;

        // Create kernel instance - for now use a simple implementation
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>, // Default fallback
        };

        // Robust SVM training using gradient descent
        let mut n_iter = 0;
        for iteration in 0..self.config.max_iter {
            n_iter = iteration + 1;
            let mut converged = true;

            for i in 0..n_samples {
                // Compute prediction
                let mut prediction = if self.config.fit_intercept {
                    intercept
                } else {
                    0.0
                };
                for j in 0..n_samples {
                    if alpha[j].abs() > 1e-10 {
                        let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                        prediction += alpha[j] * k_val;
                    }
                }

                // Compute prediction error
                let error = prediction - y[i];

                // Compute loss derivative
                let loss_derivative = self.config.loss.derivative(error);

                // Update alpha using gradient descent
                let old_alpha = alpha[i];
                let gradient = loss_derivative + alpha[i] / self.config.c;
                alpha[i] -= current_lr * gradient;

                // Update intercept if needed
                if self.config.fit_intercept {
                    intercept -= current_lr * loss_derivative;
                }

                // Check convergence
                if (alpha[i] - old_alpha).abs() > self.config.tol {
                    converged = false;
                }
            }

            // Update learning rate
            current_lr =
                (current_lr * self.config.learning_rate_decay).max(self.config.min_learning_rate);

            if converged {
                break;
            }
        }

        // Filter out non-support vectors (alpha close to zero)
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &a)| a.abs() > 1e-8)
            .map(|(i, _)| i)
            .collect();

        let final_support_vectors = if support_indices.is_empty() {
            // If no support vectors, keep the first sample
            let mut sv = Array2::zeros((1, n_features));
            sv.row_mut(0).assign(&x.row(0));
            sv
        } else {
            let mut sv = Array2::zeros((support_indices.len(), n_features));
            for (i, &idx) in support_indices.iter().enumerate() {
                sv.row_mut(i).assign(&x.row(idx));
            }
            sv
        };

        let final_alpha = if support_indices.is_empty() {
            Array1::from_vec(vec![1e-8])
        } else {
            Array1::from_vec(support_indices.iter().map(|&i| alpha[i]).collect())
        };

        Ok(RobustSVM {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(final_support_vectors),
            alpha_: Some(final_alpha),
            intercept_: Some(intercept),
            n_features_in_: Some(n_features),
            n_iter_: Some(n_iter),
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for RobustSVM<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(
                "Feature mismatch: X has different number of features than training data"
                    .to_string(),
            ));
        }

        let support_vectors = self.support_vectors_.as_ref().unwrap();
        let alpha = self.alpha_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>, // Default fallback
        };

        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut prediction = if self.config.fit_intercept {
                intercept
            } else {
                0.0
            };

            for j in 0..support_vectors.nrows() {
                let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                prediction += alpha[j] * k_val;
            }

            predictions[i] = prediction;
        }

        Ok(predictions)
    }
}

impl RobustSVM<Trained> {
    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_.as_ref().unwrap()
    }

    /// Get the support vector coefficients (alpha values)
    pub fn alpha(&self) -> &Array1<Float> {
        self.alpha_.as_ref().unwrap()
    }

    /// Get the intercept term
    pub fn intercept(&self) -> Float {
        self.intercept_.unwrap()
    }

    /// Get the number of features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get the number of iterations performed during training
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Compute the decision function values
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        self.predict(x)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_robust_svm_creation() {
        let rsvm = RobustSVM::new()
            .c(2.0)
            .loss(RobustLoss::Huber { delta: 1.5 })
            .kernel(KernelType::Linear)
            .tol(1e-4)
            .max_iter(500)
            .random_state(42);

        assert_eq!(rsvm.config.c, 2.0);
        assert_eq!(rsvm.config.loss, RobustLoss::Huber { delta: 1.5 });
        assert_eq!(rsvm.config.tol, 1e-4);
        assert_eq!(rsvm.config.max_iter, 500);
        assert_eq!(rsvm.config.random_state, Some(42));
    }

    #[test]
    fn test_huber_loss_computation() {
        let loss = RobustLoss::Huber { delta: 1.0 };

        // Test quadratic region (|error| <= delta)
        assert!((loss.loss(0.5) - 0.125).abs() < 1e-6); // 0.5 * 0.5^2
        assert!((loss.derivative(0.5) - 0.5).abs() < 1e-6);

        // Test linear region (|error| > delta)
        assert!((loss.loss(2.0) - 1.5).abs() < 1e-6); // 1.0 * (2.0 - 0.5)
        assert!((loss.derivative(2.0) - 1.0).abs() < 1e-6);
        assert!((loss.derivative(-2.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_epsilon_insensitive_loss() {
        let loss = RobustLoss::EpsilonInsensitive { epsilon: 0.1 };

        // Test inside epsilon tube
        assert!((loss.loss(0.05) - 0.0).abs() < 1e-6);
        assert!((loss.derivative(0.05) - 0.0).abs() < 1e-6);

        // Test outside epsilon tube
        assert!((loss.loss(0.3) - 0.2).abs() < 1e-6); // |0.3| - 0.1
        assert!((loss.derivative(0.3) - 1.0).abs() < 1e-6);
        assert!((loss.derivative(-0.3) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    #[ignore = "Slow test: trains robust SVM. Run with --ignored flag"]
    fn test_robust_svm_training() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ];
        let y = array![1.0, 2.0, 3.0, 6.0, 7.0, 8.0];

        let rsvm = RobustSVM::new()
            .c(1.0)
            .loss(RobustLoss::Huber { delta: 1.0 })
            .kernel(KernelType::Linear)
            .tol(1e-2)
            .max_iter(100)
            .learning_rate(0.01)
            .random_state(42);

        let fitted_model = rsvm.fit(&x, &y).unwrap();

        assert_eq!(fitted_model.n_features_in(), 2);
        assert!(fitted_model.n_iter() > 0);

        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // Predictions should be finite
        for &pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_robust_svm_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0]; // Wrong length

        let rsvm = RobustSVM::new();
        let result = rsvm.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_pinball_loss() {
        let loss = RobustLoss::Pinball { tau: 0.5 };

        // Test positive error
        assert!((loss.loss(2.0) - 1.0).abs() < 1e-6); // 0.5 * 2.0
        assert!((loss.derivative(2.0) - 0.5).abs() < 1e-6);

        // Test negative error
        assert!((loss.loss(-2.0) - 1.0).abs() < 1e-6); // (0.5 - 1.0) * (-2.0)
        assert!((loss.derivative(-2.0) - (-0.5)).abs() < 1e-6);
    }
}
