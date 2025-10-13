//! Regularized Ensemble Methods
//!
//! This module provides ensemble methods with L1/L2 regularization for ensemble weights,
//! dropout techniques for robustness, and advanced weight optimization strategies.

use crate::gradient_boosting::{GradientBoostingRegressor, TrainedGradientBoostingRegressor};
// ❌ REMOVED: rand_chacha dependencies (SciRS2 Policy violations)
// ❌ rand_chacha::rand_core - use scirs2_core::random instead
// ❌ rand_chacha::scirs2_core::random::rngs::StdRng - use scirs2_core::random instead
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit},
};

/// Helper function to generate random f64 from scirs2_core::random::RngCore
fn gen_f64(rng: &mut impl scirs2_core::random::RngCore) -> f64 {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    f64::from_le_bytes(bytes) / f64::from_le_bytes([255u8; 8])
}

/// Helper function to generate random value in range from scirs2_core::random::RngCore
fn gen_range_usize(
    rng: &mut impl scirs2_core::random::RngCore,
    range: std::ops::Range<usize>,
) -> usize {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    let val = u64::from_le_bytes(bytes);
    range.start + (val as usize % (range.end - range.start))
}

// Use concrete type instead of trait object for simplicity
type BoxedEstimator = Box<TrainedGradientBoostingRegressor>;

/// Configuration for regularized ensemble methods
#[derive(Debug, Clone)]
pub struct RegularizedEnsembleConfig {
    /// Number of base estimators
    pub n_estimators: usize,
    /// L1 regularization strength (Lasso)
    pub alpha_l1: f64,
    /// L2 regularization strength (Ridge)
    pub alpha_l2: f64,
    /// Elastic net mixing parameter (0 = Ridge, 1 = Lasso)
    pub l1_ratio: f64,
    /// Weight optimization algorithm
    pub weight_optimizer: WeightOptimizer,
    /// Dropout probability for ensemble training
    pub dropout_probability: f64,
    /// Whether to use noise injection for robustness
    pub noise_injection: bool,
    /// Noise variance for injection
    pub noise_variance: f64,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance for weight optimization
    pub tolerance: f64,
    /// Learning rate for gradient-based optimization
    pub learning_rate: f64,
    /// Weight decay factor
    pub weight_decay: f64,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for RegularizedEnsembleConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            alpha_l1: 0.1,
            alpha_l2: 0.1,
            l1_ratio: 0.5,
            weight_optimizer: WeightOptimizer::CoordinateDescent,
            dropout_probability: 0.1,
            noise_injection: false,
            noise_variance: 0.01,
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            weight_decay: 0.0001,
            random_state: None,
        }
    }
}

/// Weight optimization algorithms for ensemble weights
#[derive(Debug, Clone, PartialEq)]
pub enum WeightOptimizer {
    /// Coordinate descent for elastic net
    CoordinateDescent,
    /// Stochastic gradient descent
    SGD,
    /// Adam optimizer
    Adam,
    /// RMSprop optimizer
    RMSprop,
    /// Limited-memory BFGS
    LBFGS,
    /// Proximal gradient method
    ProximalGradient,
    /// Alternating direction method of multipliers
    ADMM,
}

/// Regularized ensemble classifier with L1/L2 regularization
pub struct RegularizedEnsembleClassifier {
    config: RegularizedEnsembleConfig,
    base_estimators: Vec<BoxedEstimator>,
    ensemble_weights: Vec<f64>,
    feature_weights: Option<Vec<f64>>,
    optimizer_state: OptimizerState,
    regularization_path: Vec<RegularizationStep>,
    is_fitted: bool,
}

/// Regularized ensemble regressor with L1/L2 regularization
pub struct RegularizedEnsembleRegressor {
    config: RegularizedEnsembleConfig,
    base_estimators: Vec<BoxedEstimator>,
    ensemble_weights: Vec<f64>,
    feature_weights: Option<Vec<f64>>,
    optimizer_state: OptimizerState,
    regularization_path: Vec<RegularizationStep>,
    is_fitted: bool,
}

/// State for optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Momentum terms for optimizers that use them
    pub momentum: Vec<f64>,
    /// Second moment estimates (for Adam)
    pub variance: Vec<f64>,
    /// Iteration counter
    pub iteration: usize,
    /// Current loss value
    pub current_loss: f64,
    /// Previous loss value for convergence checking
    pub previous_loss: f64,
    /// Gradient history for L-BFGS
    pub gradient_history: Vec<Vec<f64>>,
    /// Weight history for L-BFGS
    pub weight_history: Vec<Vec<f64>>,
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self {
            momentum: Vec::new(),
            variance: Vec::new(),
            iteration: 0,
            current_loss: f64::INFINITY,
            previous_loss: f64::INFINITY,
            gradient_history: Vec::new(),
            weight_history: Vec::new(),
        }
    }
}

/// Information about each regularization step
#[derive(Debug, Clone)]
pub struct RegularizationStep {
    /// Alpha value used in this step
    pub alpha: f64,
    /// L1 ratio used in this step
    pub l1_ratio: f64,
    /// Weights at this step
    pub weights: Vec<f64>,
    /// Loss value at this step
    pub loss: f64,
    /// Number of non-zero weights (sparsity)
    pub n_nonzero: usize,
}

/// Dropout ensemble for robustness training
pub struct DropoutEnsemble {
    config: RegularizedEnsembleConfig,
    estimators: Vec<BoxedEstimator>,
    dropout_masks: Vec<Vec<bool>>,
    rng: scirs2_core::random::CoreRandom<scirs2_core::random::rngs::StdRng>,
}

impl RegularizedEnsembleConfig {
    pub fn builder() -> RegularizedEnsembleConfigBuilder {
        RegularizedEnsembleConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct RegularizedEnsembleConfigBuilder {
    config: RegularizedEnsembleConfig,
}

impl RegularizedEnsembleConfigBuilder {
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    pub fn alpha_l1(mut self, alpha: f64) -> Self {
        self.config.alpha_l1 = alpha;
        self
    }

    pub fn alpha_l2(mut self, alpha: f64) -> Self {
        self.config.alpha_l2 = alpha;
        self
    }

    pub fn l1_ratio(mut self, ratio: f64) -> Self {
        self.config.l1_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    pub fn weight_optimizer(mut self, optimizer: WeightOptimizer) -> Self {
        self.config.weight_optimizer = optimizer;
        self
    }

    pub fn dropout_probability(mut self, prob: f64) -> Self {
        self.config.dropout_probability = prob.clamp(0.0, 1.0);
        self
    }

    pub fn noise_injection(mut self, inject: bool) -> Self {
        self.config.noise_injection = inject;
        self
    }

    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.config.max_iterations = max_iter;
        self
    }

    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.tolerance = tol;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    pub fn build(self) -> RegularizedEnsembleConfig {
        self.config
    }
}

impl RegularizedEnsembleRegressor {
    pub fn new(config: RegularizedEnsembleConfig) -> Self {
        Self {
            config,
            base_estimators: Vec::new(),
            ensemble_weights: Vec::new(),
            feature_weights: None,
            optimizer_state: OptimizerState::default(),
            regularization_path: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn builder() -> RegularizedEnsembleRegressorBuilder {
        RegularizedEnsembleRegressorBuilder::new()
    }

    /// Train base estimators
    fn train_base_estimators(&mut self, X: &Array2<f64>, y: &Vec<f64>) -> SklResult<()> {
        self.base_estimators.clear();

        for i in 0..self.config.n_estimators {
            // Add noise injection if enabled
            let (X_train, y_train) = if self.config.noise_injection {
                self.inject_noise(X, y, i)?
            } else {
                (X.clone(), y.clone())
            };

            // Create and train base estimator
            let y_train_array = Array1::from_vec(y_train.clone());
            let estimator = GradientBoostingRegressor::builder()
                .n_estimators(50)
                .learning_rate(0.1)
                .max_depth(4)
                .build()
                .fit(&X_train, &y_train_array)?;

            self.base_estimators.push(Box::new(estimator));
        }

        // Initialize weights uniformly
        self.ensemble_weights =
            vec![1.0 / self.config.n_estimators as f64; self.config.n_estimators];

        Ok(())
    }

    /// Inject noise into training data for robustness
    fn inject_noise(
        &self,
        X: &Array2<f64>,
        y: &Vec<f64>,
        seed_offset: usize,
    ) -> SklResult<(Array2<f64>, Vec<f64>)> {
        let seed = self.config.random_state.unwrap_or(42) + seed_offset as u64;
        let mut rng = scirs2_core::random::seeded_rng(seed);

        let shape = X.shape();
        let (n_samples, n_features) = (shape[0], shape[1]);
        let noise_std = self.config.noise_variance.sqrt();

        let mut X_noisy = X.clone();
        let mut y_noisy = y.clone();

        // Add Gaussian noise to features
        for i in 0..n_samples {
            for j in 0..n_features {
                let noise = gen_f64(&mut rng) * noise_std;
                X_noisy[[i, j]] += noise;
            }

            // Add noise to targets
            let target_noise = gen_f64(&mut rng) * noise_std * 0.1; // Smaller noise for targets
            y_noisy[i] += target_noise;
        }

        Ok((X_noisy, y_noisy))
    }

    /// Optimize ensemble weights with regularization
    fn optimize_ensemble_weights(&mut self, X: &Array2<f64>, y: &Vec<f64>) -> SklResult<()> {
        // Get predictions from all base estimators
        let base_predictions = self.get_base_predictions(X)?;

        // Initialize optimizer state
        self.optimizer_state = OptimizerState::default();
        self.optimizer_state.momentum = vec![0.0; self.config.n_estimators];
        self.optimizer_state.variance = vec![0.0; self.config.n_estimators];

        // Optimize weights based on selected algorithm
        match self.config.weight_optimizer {
            WeightOptimizer::CoordinateDescent => {
                self.coordinate_descent_optimization(&base_predictions, y)?;
            }
            WeightOptimizer::SGD => {
                self.sgd_optimization(&base_predictions, y)?;
            }
            WeightOptimizer::Adam => {
                self.adam_optimization(&base_predictions, y)?;
            }
            WeightOptimizer::ProximalGradient => {
                self.proximal_gradient_optimization(&base_predictions, y)?;
            }
            _ => {
                // Default to coordinate descent
                self.coordinate_descent_optimization(&base_predictions, y)?;
            }
        }

        Ok(())
    }

    /// Get predictions from all base estimators
    fn get_base_predictions(&self, X: &Array2<f64>) -> SklResult<Vec<Vec<f64>>> {
        let mut predictions = Vec::new();

        for estimator in &self.base_estimators {
            let pred = estimator.predict(X)?;
            predictions.push(pred.to_vec());
        }

        Ok(predictions)
    }

    /// Coordinate descent optimization for elastic net
    fn coordinate_descent_optimization(
        &mut self,
        predictions: &[Vec<f64>],
        y: &[f64],
    ) -> SklResult<()> {
        let n_samples = y.len();
        let n_estimators = self.config.n_estimators;

        for iteration in 0..self.config.max_iterations {
            let mut max_weight_change: f64 = 0.0;

            for j in 0..n_estimators {
                let old_weight = self.ensemble_weights[j];

                // Compute partial residual
                let mut partial_residual = 0.0;
                for i in 0..n_samples {
                    let mut ensemble_pred = 0.0;
                    for k in 0..n_estimators {
                        if k != j {
                            ensemble_pred += self.ensemble_weights[k] * predictions[k][i];
                        }
                    }
                    partial_residual += predictions[j][i] * (y[i] - ensemble_pred);
                }

                // Compute L2 norm squared for this estimator
                let mut l2_norm_sq = 0.0;
                for i in 0..n_samples {
                    l2_norm_sq += predictions[j][i] * predictions[j][i];
                }

                // Soft thresholding for L1 regularization
                let l1_penalty = self.config.alpha_l1 * self.config.l1_ratio;
                let l2_penalty = self.config.alpha_l2 * (1.0 - self.config.l1_ratio);

                let numerator =
                    self.soft_threshold(partial_residual, l1_penalty * n_samples as f64);
                let denominator = l2_norm_sq + l2_penalty * n_samples as f64;

                self.ensemble_weights[j] = if denominator > 0.0 {
                    numerator / denominator
                } else {
                    0.0
                };

                let weight_change = (self.ensemble_weights[j] - old_weight).abs();
                max_weight_change = max_weight_change.max(weight_change);
            }

            // Check convergence
            if max_weight_change < self.config.tolerance {
                break;
            }

            // Record regularization step
            let loss = self.compute_loss(predictions, y);
            let n_nonzero = self
                .ensemble_weights
                .iter()
                .filter(|&&w| w.abs() > 1e-10)
                .count();

            self.regularization_path.push(RegularizationStep {
                alpha: self.config.alpha_l1 + self.config.alpha_l2,
                l1_ratio: self.config.l1_ratio,
                weights: self.ensemble_weights.clone(),
                loss,
                n_nonzero,
            });
        }

        Ok(())
    }

    /// Stochastic gradient descent optimization
    fn sgd_optimization(&mut self, predictions: &[Vec<f64>], y: &[f64]) -> SklResult<()> {
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        let n_samples = y.len();
        let n_estimators = self.config.n_estimators;

        for iteration in 0..self.config.max_iterations {
            // Randomly sample a batch
            let sample_idx = gen_range_usize(&mut rng, 0..n_samples);

            // Compute prediction for this sample
            let mut pred = 0.0;
            for j in 0..n_estimators {
                pred += self.ensemble_weights[j] * predictions[j][sample_idx];
            }

            let error = pred - y[sample_idx];

            // Compute gradients and update weights
            for j in 0..n_estimators {
                let gradient = error * predictions[j][sample_idx];

                // Add L1 and L2 regularization terms
                let l1_grad =
                    self.config.alpha_l1 * self.config.l1_ratio * self.ensemble_weights[j].signum();
                let l2_grad =
                    self.config.alpha_l2 * (1.0 - self.config.l1_ratio) * self.ensemble_weights[j];

                let total_gradient = gradient + l1_grad + l2_grad;

                // Update with momentum
                self.optimizer_state.momentum[j] = 0.9 * self.optimizer_state.momentum[j]
                    + self.config.learning_rate * total_gradient;
                self.ensemble_weights[j] -= self.optimizer_state.momentum[j];
            }

            // Apply weight decay
            for weight in &mut self.ensemble_weights {
                *weight *= 1.0 - self.config.weight_decay;
            }

            // Check convergence periodically
            if iteration % 100 == 0 {
                let loss = self.compute_loss(predictions, y);
                if (self.optimizer_state.previous_loss - loss).abs() < self.config.tolerance {
                    break;
                }
                self.optimizer_state.previous_loss = loss;
            }
        }

        Ok(())
    }

    /// Adam optimization algorithm
    fn adam_optimization(&mut self, predictions: &[Vec<f64>], y: &[f64]) -> SklResult<()> {
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        let n_samples = y.len();
        let n_estimators = self.config.n_estimators;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        for iteration in 0..self.config.max_iterations {
            // Randomly sample a batch
            let sample_idx = gen_range_usize(&mut rng, 0..n_samples);

            // Compute prediction for this sample
            let mut pred = 0.0;
            for j in 0..n_estimators {
                pred += self.ensemble_weights[j] * predictions[j][sample_idx];
            }

            let error = pred - y[sample_idx];

            // Compute gradients and update weights using Adam
            for j in 0..n_estimators {
                let gradient = error * predictions[j][sample_idx];

                // Add regularization terms
                let l1_grad =
                    self.config.alpha_l1 * self.config.l1_ratio * self.ensemble_weights[j].signum();
                let l2_grad =
                    self.config.alpha_l2 * (1.0 - self.config.l1_ratio) * self.ensemble_weights[j];

                let total_gradient = gradient + l1_grad + l2_grad;

                // Update biased first moment estimate
                self.optimizer_state.momentum[j] =
                    beta1 * self.optimizer_state.momentum[j] + (1.0 - beta1) * total_gradient;

                // Update biased second raw moment estimate
                self.optimizer_state.variance[j] = beta2 * self.optimizer_state.variance[j]
                    + (1.0 - beta2) * total_gradient * total_gradient;

                // Compute bias-corrected first and second moment estimates
                let m_hat =
                    self.optimizer_state.momentum[j] / (1.0 - beta1.powi((iteration + 1) as i32));
                let v_hat =
                    self.optimizer_state.variance[j] / (1.0 - beta2.powi((iteration + 1) as i32));

                // Update weights
                self.ensemble_weights[j] -=
                    self.config.learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            }

            // Check convergence periodically
            if iteration % 100 == 0 {
                let loss = self.compute_loss(predictions, y);
                if (self.optimizer_state.previous_loss - loss).abs() < self.config.tolerance {
                    break;
                }
                self.optimizer_state.previous_loss = loss;
            }
        }

        Ok(())
    }

    /// Proximal gradient optimization
    fn proximal_gradient_optimization(
        &mut self,
        predictions: &[Vec<f64>],
        y: &[f64],
    ) -> SklResult<()> {
        let n_samples = y.len();
        let n_estimators = self.config.n_estimators;

        for _iteration in 0..self.config.max_iterations {
            // Compute full gradient
            let mut gradients = vec![0.0; n_estimators];

            for i in 0..n_samples {
                let mut pred = 0.0;
                for j in 0..n_estimators {
                    pred += self.ensemble_weights[j] * predictions[j][i];
                }

                let error = pred - y[i];

                for j in 0..n_estimators {
                    gradients[j] += error * predictions[j][i] / n_samples as f64;
                }
            }

            // Gradient step
            for j in 0..n_estimators {
                self.ensemble_weights[j] -= self.config.learning_rate * gradients[j];
            }

            // Proximal operator for L1 regularization
            let threshold = self.config.learning_rate * self.config.alpha_l1 * self.config.l1_ratio;
            for weight in &mut self.ensemble_weights {
                let original_weight = *weight;
                *weight = Self::soft_threshold_static(original_weight, threshold);
            }

            // L2 regularization (closed form)
            let l2_shrinkage = 1.0
                / (1.0
                    + self.config.learning_rate
                        * self.config.alpha_l2
                        * (1.0 - self.config.l1_ratio));
            for weight in &mut self.ensemble_weights {
                *weight *= l2_shrinkage;
            }
        }

        Ok(())
    }

    /// Soft thresholding operator for L1 regularization
    fn soft_threshold(&self, x: f64, threshold: f64) -> f64 {
        Self::soft_threshold_static(x, threshold)
    }

    /// Static version of soft thresholding operator for L1 regularization
    fn soft_threshold_static(x: f64, threshold: f64) -> f64 {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }

    /// Compute loss function (MSE + regularization)
    fn compute_loss(&self, predictions: &[Vec<f64>], y: &[f64]) -> f64 {
        let n_samples = y.len();
        let mut mse = 0.0;

        for i in 0..n_samples {
            let mut pred = 0.0;
            for j in 0..self.config.n_estimators {
                pred += self.ensemble_weights[j] * predictions[j][i];
            }
            let error = pred - y[i];
            mse += error * error;
        }
        mse /= n_samples as f64;

        // Add regularization terms
        let l1_penalty = self.config.alpha_l1
            * self.config.l1_ratio
            * self.ensemble_weights.iter().map(|&w| w.abs()).sum::<f64>();
        let l2_penalty = self.config.alpha_l2
            * (1.0 - self.config.l1_ratio)
            * self.ensemble_weights.iter().map(|&w| w * w).sum::<f64>()
            / 2.0;

        mse + l1_penalty + l2_penalty
    }

    /// Get ensemble weights
    pub fn get_ensemble_weights(&self) -> &[f64] {
        &self.ensemble_weights
    }

    /// Get regularization path
    pub fn get_regularization_path(&self) -> &[RegularizationStep] {
        &self.regularization_path
    }

    /// Get sparsity level (fraction of zero weights)
    pub fn get_sparsity(&self) -> f64 {
        let n_nonzero = self
            .ensemble_weights
            .iter()
            .filter(|&&w| w.abs() > 1e-10)
            .count();
        1.0 - (n_nonzero as f64 / self.ensemble_weights.len() as f64)
    }
}

impl DropoutEnsemble {
    pub fn new(config: RegularizedEnsembleConfig) -> Self {
        let rng = if let Some(seed) = config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        Self {
            config,
            estimators: Vec::new(),
            dropout_masks: Vec::new(),
            rng,
        }
    }

    /// Train ensemble with dropout
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &Array2<f64>, y: &Vec<f64>) -> SklResult<()> {
        self.estimators.clear();
        self.dropout_masks.clear();

        for _ in 0..self.config.n_estimators {
            // Generate dropout mask
            let mut mask = Vec::new();
            for _ in 0..X.shape()[1] {
                mask.push(gen_f64(&mut self.rng) > self.config.dropout_probability);
            }

            // Apply dropout to features
            let X_dropout = self.apply_dropout_mask(X, &mask)?;

            // Train estimator on dropout data
            let y_array = Array1::from_vec(y.clone());
            let estimator = GradientBoostingRegressor::builder()
                .n_estimators(50)
                .learning_rate(0.1)
                .max_depth(4)
                .build()
                .fit(&X_dropout, &y_array)?;

            self.estimators.push(Box::new(estimator));
            self.dropout_masks.push(mask);
        }

        Ok(())
    }

    /// Apply dropout mask to features
    fn apply_dropout_mask(&self, X: &Array2<f64>, mask: &[bool]) -> SklResult<Array2<f64>> {
        let shape = X.shape();
        let (n_samples, n_features) = (shape[0], shape[1]);
        let mut X_dropout = X.clone();

        for i in 0..n_samples {
            for j in 0..n_features {
                if !mask[j] {
                    X_dropout[[i, j]] = 0.0;
                } else {
                    // Scale by dropout probability to maintain expected values
                    X_dropout[[i, j]] /= 1.0 - self.config.dropout_probability;
                }
            }
        }

        Ok(X_dropout)
    }

    /// Predict with dropout ensemble
    #[allow(non_snake_case)]
    pub fn predict(&self, X: &Array2<f64>) -> SklResult<Vec<f64>> {
        if self.estimators.is_empty() {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let n_samples = X.shape()[0];
        let mut predictions = vec![0.0; n_samples];

        for (estimator, mask) in self.estimators.iter().zip(self.dropout_masks.iter()) {
            let X_masked = self.apply_dropout_mask(X, mask)?;
            let pred = estimator.predict(&X_masked)?;

            for (i, &p) in pred.iter().enumerate() {
                predictions[i] += p;
            }
        }

        // Average predictions
        for p in &mut predictions {
            *p /= self.estimators.len() as f64;
        }

        Ok(predictions)
    }
}

pub struct RegularizedEnsembleRegressorBuilder {
    config: RegularizedEnsembleConfig,
}

impl Default for RegularizedEnsembleRegressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RegularizedEnsembleRegressorBuilder {
    pub fn new() -> Self {
        Self {
            config: RegularizedEnsembleConfig::default(),
        }
    }

    pub fn config(mut self, config: RegularizedEnsembleConfig) -> Self {
        self.config = config;
        self
    }

    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    pub fn alpha_l1(mut self, alpha: f64) -> Self {
        self.config.alpha_l1 = alpha;
        self
    }

    pub fn alpha_l2(mut self, alpha: f64) -> Self {
        self.config.alpha_l2 = alpha;
        self
    }

    pub fn weight_optimizer(mut self, optimizer: WeightOptimizer) -> Self {
        self.config.weight_optimizer = optimizer;
        self
    }

    pub fn build(self) -> RegularizedEnsembleRegressor {
        RegularizedEnsembleRegressor::new(self.config)
    }
}

impl Estimator for RegularizedEnsembleRegressor {
    type Config = RegularizedEnsembleConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Vec<f64>> for RegularizedEnsembleRegressor {
    type Fitted = Self;

    fn fit(mut self, X: &Array2<f64>, y: &Vec<f64>) -> SklResult<Self::Fitted> {
        // Train base estimators
        self.train_base_estimators(X, y)?;

        // Optimize ensemble weights with regularization
        self.optimize_ensemble_weights(X, y)?;

        self.is_fitted = true;
        Ok(self)
    }
}

impl Predict<Array2<f64>, Vec<f64>> for RegularizedEnsembleRegressor {
    fn predict(&self, X: &Array2<f64>) -> SklResult<Vec<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let base_predictions = self.get_base_predictions(X)?;
        let n_samples = X.shape()[0];
        let mut predictions = vec![0.0; n_samples];

        for i in 0..n_samples {
            for j in 0..self.config.n_estimators {
                predictions[i] += self.ensemble_weights[j] * base_predictions[j][i];
            }
        }

        Ok(predictions)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_regularized_config() {
        let config = RegularizedEnsembleConfig::builder()
            .n_estimators(5)
            .alpha_l1(0.1)
            .alpha_l2(0.2)
            .l1_ratio(0.7)
            .build();

        assert_eq!(config.n_estimators, 5);
        assert_eq!(config.alpha_l1, 0.1);
        assert_eq!(config.alpha_l2, 0.2);
        assert_eq!(config.l1_ratio, 0.7);
    }

    #[test]
    fn test_soft_threshold() {
        let config = RegularizedEnsembleConfig::default();
        let ensemble = RegularizedEnsembleRegressor::new(config);

        assert_eq!(ensemble.soft_threshold(1.0, 0.5), 0.5);
        assert_eq!(ensemble.soft_threshold(-1.0, 0.5), -0.5);
        assert_eq!(ensemble.soft_threshold(0.3, 0.5), 0.0);
        assert_eq!(ensemble.soft_threshold(-0.3, 0.5), 0.0);
    }

    #[test]
    fn test_regularized_ensemble_basic() {
        let config = RegularizedEnsembleConfig::builder()
            .n_estimators(3)
            .alpha_l1(0.1)
            .alpha_l2(0.1)
            .max_iterations(10)
            .random_state(42)
            .build();

        let ensemble = RegularizedEnsembleRegressor::new(config);

        // Test basic configuration
        assert_eq!(ensemble.config.n_estimators, 3);
        assert_eq!(ensemble.config.alpha_l1, 0.1);
        assert_eq!(ensemble.config.alpha_l2, 0.1);
        assert!(!ensemble.is_fitted);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dropout_ensemble() {
        let config = RegularizedEnsembleConfig::builder()
            .n_estimators(3)
            .dropout_probability(0.2)
            .random_state(42)
            .build();

        let mut dropout_ensemble = DropoutEnsemble::new(config);

        let X = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = vec![1.0, 2.0, 3.0];

        dropout_ensemble.fit(&X, &y).unwrap();

        let X_test = Array2::from_shape_vec((1, 2), vec![7.0, 8.0]).unwrap();

        let predictions = dropout_ensemble.predict(&X_test).unwrap();
        assert_eq!(predictions.len(), 1);
    }

    #[test]
    fn test_sparsity_calculation() {
        let config = RegularizedEnsembleConfig::default();
        let mut ensemble = RegularizedEnsembleRegressor::new(config);

        // Set some weights to zero
        ensemble.ensemble_weights = vec![0.5, 0.0, 0.3, 0.0, 0.2];

        let sparsity = ensemble.get_sparsity();
        assert_eq!(sparsity, 0.4); // 2 out of 5 weights are zero
    }
}
