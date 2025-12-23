//! Joint Loss Optimization for Multi-Output Learning
//!
//! This module provides joint loss optimization techniques for multi-output learning problems,
//! where multiple output losses are combined using various strategies to optimize all outputs
//! simultaneously rather than independently.
//!
//! ## Key Features
//!
//! - **Multiple Loss Functions**: Support for MSE, MAE, Huber, Cross-entropy, and Hinge losses
//! - **Flexible Loss Combination**: Sum, weighted sum, max, geometric mean, and adaptive strategies
//! - **Gradient-Based Optimization**: Efficient gradient computation for joint loss minimization
//! - **Regularization**: L2 regularization to prevent overfitting
//! - **Configurable Training**: Customizable learning rate, iterations, and convergence criteria

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::thread_rng;
use scirs2_core::random::RandNormal;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Loss function types for joint optimization
#[derive(Debug, Clone, PartialEq)]
pub enum LossFunction {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Huber Loss with configurable delta
    Huber(Float),
    /// Cross-entropy loss
    CrossEntropy,
    /// Hinge loss
    Hinge,
    /// Custom loss function
    Custom(String),
}

/// Loss combination strategies for joint optimization
#[derive(Debug, Clone, PartialEq)]
pub enum LossCombination {
    /// Simple sum of individual losses
    Sum,
    /// Weighted sum of individual losses
    WeightedSum(Vec<Float>),
    /// Maximum of individual losses
    Max,
    /// Geometric mean of individual losses
    GeometricMean,
    /// Adaptive weighting based on loss magnitudes
    Adaptive,
}

/// Joint Loss Optimization configuration
#[derive(Debug, Clone)]
pub struct JointLossConfig {
    /// Loss function for each output
    pub output_losses: Vec<LossFunction>,
    /// Loss combination strategy
    pub combination: LossCombination,
    /// Regularization strength
    pub regularization: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Learning rate
    pub learning_rate: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for JointLossConfig {
    fn default() -> Self {
        Self {
            output_losses: vec![LossFunction::MSE],
            combination: LossCombination::Sum,
            regularization: 0.01,
            max_iter: 1000,
            tol: 1e-6,
            learning_rate: 0.01,
            random_state: None,
        }
    }
}

/// Joint Loss Optimizer for multi-output learning
#[derive(Debug, Clone)]
pub struct JointLossOptimizer<S = Untrained> {
    state: S,
    config: JointLossConfig,
}

/// Trained state for Joint Loss Optimizer
#[derive(Debug, Clone)]
pub struct JointLossOptimizerTrained {
    /// Model weights
    pub weights: Array2<Float>,
    /// Bias terms
    pub bias: Array1<Float>,
    /// Number of features
    pub n_features: usize,
    /// Number of outputs
    pub n_outputs: usize,
    /// Training history
    pub loss_history: Vec<Float>,
    /// Configuration used for training
    pub config: JointLossConfig,
}

impl JointLossOptimizer<Untrained> {
    /// Create a new Joint Loss Optimizer
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: JointLossConfig::default(),
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: JointLossConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the output loss functions
    pub fn output_losses(mut self, losses: Vec<LossFunction>) -> Self {
        self.config.output_losses = losses;
        self
    }

    /// Set the loss combination strategy
    pub fn combination(mut self, combination: LossCombination) -> Self {
        self.config.combination = combination;
        self
    }

    /// Set the regularization strength
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.config.regularization = regularization;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

impl Default for JointLossOptimizer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for JointLossOptimizer<Untrained> {
    type Config = JointLossConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for JointLossOptimizer<Untrained> {
    type Fitted = JointLossOptimizer<JointLossOptimizerTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView2<'_, Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let (y_samples, n_outputs) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if n_outputs != self.config.output_losses.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of outputs ({}) must match number of loss functions ({})",
                n_outputs,
                self.config.output_losses.len()
            )));
        }

        let mut rng = thread_rng();

        // Initialize weights using Xavier initialization
        let std_dev = (2.0 / (n_features + n_outputs) as Float).sqrt();
        let normal_dist = RandNormal::new(0.0, std_dev).unwrap();
        let mut weights = Array2::<Float>::zeros((n_features, n_outputs));
        for i in 0..n_features {
            for j in 0..n_outputs {
                weights[[i, j]] = rng.sample(normal_dist);
            }
        }
        let mut bias = Array1::<Float>::zeros(n_outputs);

        let mut loss_history = Vec::new();
        let mut prev_loss = Float::INFINITY;

        for iteration in 0..self.config.max_iter {
            // Forward pass
            let predictions = X.dot(&weights) + &bias;

            // Compute joint loss
            let joint_loss = self.compute_joint_loss(&predictions, y)?;
            loss_history.push(joint_loss);

            // Check convergence
            if (prev_loss - joint_loss).abs() < self.config.tol {
                break;
            }
            prev_loss = joint_loss;

            // Compute gradients
            let (weight_gradients, bias_gradients) = self.compute_gradients(X, y, &predictions)?;

            // Update weights and bias
            weights = weights - self.config.learning_rate * weight_gradients;
            bias = bias - self.config.learning_rate * bias_gradients;

            // Apply regularization
            if self.config.regularization > 0.0 {
                weights *= 1.0 - self.config.regularization * self.config.learning_rate;
            }
        }

        Ok(JointLossOptimizer {
            state: JointLossOptimizerTrained {
                weights,
                bias,
                n_features,
                n_outputs,
                loss_history,
                config: self.config.clone(),
            },
            config: self.config,
        })
    }
}

impl JointLossOptimizer<Untrained> {
    /// Compute joint loss based on the combination strategy
    fn compute_joint_loss(
        &self,
        predictions: &Array2<Float>,
        y: &ArrayView2<'_, Float>,
    ) -> SklResult<Float> {
        let mut individual_losses = Vec::new();

        for (i, loss_fn) in self.config.output_losses.iter().enumerate() {
            let pred_col = predictions.column(i);
            let y_col = y.column(i);
            let loss = self.compute_individual_loss(loss_fn, &pred_col, &y_col)?;
            individual_losses.push(loss);
        }

        let joint_loss = match &self.config.combination {
            LossCombination::Sum => individual_losses.iter().sum(),
            LossCombination::WeightedSum(weights) => {
                if weights.len() != individual_losses.len() {
                    return Err(SklearsError::InvalidInput(
                        "Weight vector length must match number of outputs".to_string(),
                    ));
                }
                individual_losses
                    .iter()
                    .zip(weights.iter())
                    .map(|(loss, weight)| loss * weight)
                    .sum()
            }
            LossCombination::Max => individual_losses.iter().cloned().fold(0.0, Float::max),
            LossCombination::GeometricMean => {
                let product: Float = individual_losses.iter().product();
                product.powf(1.0 / individual_losses.len() as Float)
            }
            LossCombination::Adaptive => {
                // Adaptive weighting based on loss magnitudes
                let total_loss: Float = individual_losses.iter().sum();
                if total_loss > 0.0 {
                    let weights: Vec<Float> = individual_losses
                        .iter()
                        .map(|&loss| loss / total_loss)
                        .collect();
                    individual_losses
                        .iter()
                        .zip(weights.iter())
                        .map(|(loss, weight)| loss * weight)
                        .sum()
                } else {
                    0.0
                }
            }
        };

        Ok(joint_loss)
    }

    /// Compute individual loss for a specific output
    fn compute_individual_loss(
        &self,
        loss_fn: &LossFunction,
        predictions: &ArrayView1<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Float> {
        match loss_fn {
            LossFunction::MSE => {
                let diff = predictions - y;
                Ok(diff.mapv(|x| x * x).mean().unwrap_or(0.0))
            }
            LossFunction::MAE => {
                let diff = predictions - y;
                Ok(diff.mapv(|x| x.abs()).mean().unwrap_or(0.0))
            }
            LossFunction::Huber(delta) => {
                let diff = predictions - y;
                let huber_loss = diff.mapv(|x| {
                    if x.abs() <= *delta {
                        0.5 * x * x
                    } else {
                        delta * x.abs() - 0.5 * delta * delta
                    }
                });
                Ok(huber_loss.mean().unwrap_or(0.0))
            }
            LossFunction::CrossEntropy => {
                // Assuming binary classification with sigmoid activation
                let epsilon = 1e-15;
                let clipped_preds = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
                let loss = y
                    .iter()
                    .zip(clipped_preds.iter())
                    .map(|(y_true, y_pred)| {
                        -(y_true * y_pred.ln() + (1.0 - y_true) * (1.0 - y_pred).ln())
                    })
                    .sum::<Float>()
                    / y.len() as Float;
                Ok(loss)
            }
            LossFunction::Hinge => {
                let loss = predictions
                    .iter()
                    .zip(y.iter())
                    .map(|(pred, true_val)| {
                        let margin = true_val * pred;
                        if margin < 1.0 {
                            1.0 - margin
                        } else {
                            0.0
                        }
                    })
                    .sum::<Float>()
                    / y.len() as Float;
                Ok(loss)
            }
            LossFunction::Custom(_) => Err(SklearsError::InvalidInput(
                "Custom loss functions are not yet implemented".to_string(),
            )),
        }
    }

    /// Compute gradients for weights and bias
    fn compute_gradients(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        predictions: &Array2<Float>,
    ) -> SklResult<(Array2<Float>, Array1<Float>)> {
        let (n_samples, n_features) = X.dim();
        let n_outputs = y.ncols();

        let mut weight_gradients = Array2::<Float>::zeros((n_features, n_outputs));
        let mut bias_gradients = Array1::<Float>::zeros(n_outputs);

        for (i, loss_fn) in self.config.output_losses.iter().enumerate() {
            let pred_col = predictions.column(i);
            let y_col = y.column(i);

            // Compute gradient for this output
            let output_gradient = self.compute_output_gradient(loss_fn, &pred_col, &y_col)?;

            // Update weight gradients
            for j in 0..n_features {
                weight_gradients[(j, i)] = X.column(j).dot(&output_gradient) / n_samples as Float;
            }

            // Update bias gradients
            bias_gradients[i] = output_gradient.mean().unwrap_or(0.0);
        }

        Ok((weight_gradients, bias_gradients))
    }

    /// Compute gradient for a specific output
    fn compute_output_gradient(
        &self,
        loss_fn: &LossFunction,
        predictions: &ArrayView1<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<Array1<Float>> {
        let gradient = match loss_fn {
            LossFunction::MSE => 2.0 * (predictions - y),
            LossFunction::MAE => (predictions - y).mapv(|x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }),
            LossFunction::Huber(delta) => {
                let diff = predictions - y;
                diff.mapv(|x| {
                    if x.abs() <= *delta {
                        x
                    } else {
                        delta * x.signum()
                    }
                })
            }
            LossFunction::CrossEntropy => {
                // Gradient for binary cross-entropy with sigmoid
                let epsilon = 1e-15;
                let clipped_preds = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
                &clipped_preds - y
            }
            LossFunction::Hinge => predictions
                .iter()
                .zip(y.iter())
                .map(|(pred, true_val)| {
                    let margin = true_val * pred;
                    if margin < 1.0 {
                        -true_val
                    } else {
                        0.0
                    }
                })
                .collect::<Array1<Float>>(),
            LossFunction::Custom(_) => {
                return Err(SklearsError::InvalidInput(
                    "Custom loss functions are not yet implemented".to_string(),
                ));
            }
        };

        Ok(gradient)
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>>
    for JointLossOptimizer<JointLossOptimizerTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features, n_features
            )));
        }

        let predictions = X.dot(&self.state.weights) + &self.state.bias;
        Ok(predictions)
    }
}

impl Estimator for JointLossOptimizer<JointLossOptimizerTrained> {
    type Config = JointLossConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.state.config
    }
}

impl JointLossOptimizer<JointLossOptimizerTrained> {
    /// Get the training loss history
    pub fn loss_history(&self) -> &[Float] {
        &self.state.loss_history
    }

    /// Get the model weights
    pub fn weights(&self) -> &Array2<Float> {
        &self.state.weights
    }

    /// Get the bias terms
    pub fn bias(&self) -> &Array1<Float> {
        &self.state.bias
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.state.n_features
    }

    /// Get the number of outputs
    pub fn n_outputs(&self) -> usize {
        self.state.n_outputs
    }
}
