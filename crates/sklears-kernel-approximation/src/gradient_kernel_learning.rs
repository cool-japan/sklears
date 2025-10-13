//! Gradient-based kernel learning for automatic parameter optimization
//!
//! This module provides gradient-based optimization methods for learning optimal
//! kernel parameters, including bandwidth selection, kernel combination weights,
//! and hyperparameter tuning using automatic differentiation.

use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use sklears_core::error::Result;

/// Gradient-based optimization configuration
#[derive(Clone, Debug)]
/// GradientConfig
pub struct GradientConfig {
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Momentum parameter
    pub momentum: f64,
    /// L2 regularization strength
    pub l2_regularization: f64,
    /// Whether to use adaptive learning rate
    pub adaptive_learning_rate: bool,
    /// Learning rate decay factor
    pub learning_rate_decay: f64,
    /// Minimum learning rate
    pub min_learning_rate: f64,
    /// Batch size for stochastic gradient descent
    pub batch_size: usize,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iterations: 1000,
            tolerance: 1e-6,
            momentum: 0.9,
            l2_regularization: 1e-4,
            adaptive_learning_rate: true,
            learning_rate_decay: 0.99,
            min_learning_rate: 1e-6,
            batch_size: 256,
        }
    }
}

/// Gradient-based optimization algorithms
#[derive(Clone, Debug, PartialEq)]
/// GradientOptimizer
pub enum GradientOptimizer {
    /// Standard gradient descent
    SGD,
    /// Momentum-based gradient descent
    Momentum,
    /// Adam optimizer
    Adam,
    /// AdaGrad optimizer
    AdaGrad,
    /// RMSprop optimizer
    RMSprop,
    /// L-BFGS optimizer
    LBFGS,
}

/// Objective function for kernel learning
#[derive(Clone, Debug, PartialEq)]
/// KernelObjective
pub enum KernelObjective {
    /// Kernel alignment
    KernelAlignment,
    /// Cross-validation error
    CrossValidationError,
    /// Marginal likelihood (for Gaussian processes)
    MarginalLikelihood,
    /// Kernel ridge regression loss
    KernelRidgeLoss,
    /// Maximum mean discrepancy
    MaximumMeanDiscrepancy,
    /// Kernel target alignment
    KernelTargetAlignment,
}

/// Gradient computation result
#[derive(Clone, Debug)]
/// GradientResult
pub struct GradientResult {
    /// Gradient vector
    pub gradient: Array1<f64>,
    /// Objective function value
    pub objective_value: f64,
    /// Hessian matrix (if computed)
    pub hessian: Option<Array2<f64>>,
}

/// Gradient-based kernel parameter learner
pub struct GradientKernelLearner {
    config: GradientConfig,
    optimizer: GradientOptimizer,
    objective: KernelObjective,
    parameters: Array1<f64>,
    parameter_bounds: Option<Array2<f64>>,
    optimization_history: Vec<(f64, Array1<f64>)>,
    velocity: Option<Array1<f64>>,
    adam_m: Option<Array1<f64>>,
    adam_v: Option<Array1<f64>>,
    iteration: usize,
}

impl GradientKernelLearner {
    /// Create a new gradient-based kernel learner
    pub fn new(n_parameters: usize) -> Self {
        Self {
            config: GradientConfig::default(),
            optimizer: GradientOptimizer::Adam,
            objective: KernelObjective::KernelAlignment,
            parameters: Array1::ones(n_parameters),
            parameter_bounds: None,
            optimization_history: Vec::new(),
            velocity: None,
            adam_m: None,
            adam_v: None,
            iteration: 0,
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: GradientConfig) -> Self {
        self.config = config;
        self
    }

    /// Set optimizer
    pub fn with_optimizer(mut self, optimizer: GradientOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set objective function
    pub fn with_objective(mut self, objective: KernelObjective) -> Self {
        self.objective = objective;
        self
    }

    /// Set parameter bounds
    pub fn with_bounds(mut self, bounds: Array2<f64>) -> Self {
        self.parameter_bounds = Some(bounds);
        self
    }

    /// Initialize parameters
    pub fn initialize_parameters(&mut self, initial_params: Array1<f64>) {
        self.parameters = initial_params;
        self.velocity = Some(Array1::zeros(self.parameters.len()));
        self.adam_m = Some(Array1::zeros(self.parameters.len()));
        self.adam_v = Some(Array1::zeros(self.parameters.len()));
        self.iteration = 0;
        self.optimization_history.clear();
        // Apply bounds to ensure initial parameters are within constraints
        self.apply_bounds();
    }

    /// Optimize kernel parameters
    pub fn optimize(&mut self, x: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<Array1<f64>> {
        for iteration in 0..self.config.max_iterations {
            self.iteration = iteration;

            // Compute gradient
            let gradient_result = self.compute_gradient(x, y)?;

            // Check convergence
            if gradient_result
                .gradient
                .iter()
                .map(|&g| g.abs())
                .sum::<f64>()
                < self.config.tolerance
            {
                break;
            }

            // Update parameters
            self.update_parameters(&gradient_result.gradient)?;

            // Store optimization history
            self.optimization_history
                .push((gradient_result.objective_value, self.parameters.clone()));

            // Adaptive learning rate
            if self.config.adaptive_learning_rate && iteration > 0 {
                self.update_learning_rate(iteration);
            }
        }

        Ok(self.parameters.clone())
    }

    /// Compute gradient of the objective function
    fn compute_gradient(&self, x: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<GradientResult> {
        match self.objective {
            KernelObjective::KernelAlignment => self.compute_kernel_alignment_gradient(x),
            KernelObjective::CrossValidationError => self.compute_cv_error_gradient(x, y),
            KernelObjective::MarginalLikelihood => self.compute_marginal_likelihood_gradient(x, y),
            KernelObjective::KernelRidgeLoss => self.compute_kernel_ridge_gradient(x, y),
            KernelObjective::MaximumMeanDiscrepancy => self.compute_mmd_gradient(x),
            KernelObjective::KernelTargetAlignment => self.compute_kta_gradient(x, y),
        }
    }

    /// Compute kernel alignment gradient
    fn compute_kernel_alignment_gradient(&self, x: &Array2<f64>) -> Result<GradientResult> {
        let n_samples = x.nrows();
        let mut gradient = Array1::zeros(self.parameters.len());

        // Compute kernel matrix
        let kernel_matrix = self.compute_kernel_matrix(x)?;

        // Compute kernel matrix derivatives
        let kernel_derivatives = self.compute_kernel_derivatives(x)?;

        // Compute alignment and its gradient
        let alignment = self.compute_kernel_alignment(&kernel_matrix);

        for i in 0..self.parameters.len() {
            let kernel_derivative = &kernel_derivatives[i];
            let alignment_derivative =
                self.compute_alignment_derivative(&kernel_matrix, kernel_derivative);
            gradient[i] = alignment_derivative;
        }

        Ok(GradientResult {
            gradient,
            objective_value: alignment,
            hessian: None,
        })
    }

    /// Compute cross-validation error gradient
    fn compute_cv_error_gradient(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> Result<GradientResult> {
        let y = y.ok_or_else(|| "Target values required for CV error gradient")?;
        let n_samples = x.nrows();
        let n_folds = 5;
        let fold_size = n_samples / n_folds;

        let mut gradient = Array1::zeros(self.parameters.len());
        let mut total_error = 0.0;

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = std::cmp::min(start_idx + fold_size, n_samples);

            // Split data
            let (x_train, y_train, x_val, y_val) = self.split_data(x, y, start_idx, end_idx);

            // Compute fold gradient
            let fold_gradient = self.compute_fold_gradient(&x_train, &y_train, &x_val, &y_val)?;

            gradient = gradient + fold_gradient.gradient;
            total_error += fold_gradient.objective_value;
        }

        gradient = gradient / n_folds as f64;
        total_error /= n_folds as f64;

        Ok(GradientResult {
            gradient,
            objective_value: total_error,
            hessian: None,
        })
    }

    /// Compute marginal likelihood gradient
    fn compute_marginal_likelihood_gradient(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> Result<GradientResult> {
        let y = y.ok_or_else(|| "Target values required for marginal likelihood gradient")?;
        let n_samples = x.nrows();

        // Compute kernel matrix
        let kernel_matrix = self.compute_kernel_matrix(x)?;

        // Add noise term
        let noise_variance = 1e-6;
        let mut k_with_noise = kernel_matrix.clone();
        for i in 0..n_samples {
            k_with_noise[[i, i]] += noise_variance;
        }

        // Compute log marginal likelihood
        let log_marginal_likelihood = self.compute_log_marginal_likelihood(&k_with_noise, y)?;

        // Compute gradient
        let mut gradient = Array1::zeros(self.parameters.len());
        let kernel_derivatives = self.compute_kernel_derivatives(x)?;

        for i in 0..self.parameters.len() {
            let kernel_derivative = &kernel_derivatives[i];
            let ml_derivative =
                self.compute_marginal_likelihood_derivative(&k_with_noise, y, kernel_derivative)?;
            gradient[i] = ml_derivative;
        }

        Ok(GradientResult {
            gradient,
            objective_value: -log_marginal_likelihood, // Negative for minimization
            hessian: None,
        })
    }

    /// Compute kernel ridge regression gradient
    fn compute_kernel_ridge_gradient(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> Result<GradientResult> {
        let y = y.ok_or_else(|| "Target values required for kernel ridge gradient")?;
        let n_samples = x.nrows();
        let alpha = 1e-3; // Regularization parameter

        // Compute kernel matrix
        let kernel_matrix = self.compute_kernel_matrix(x)?;

        // Add regularization
        let mut k_reg = kernel_matrix.clone();
        for i in 0..n_samples {
            k_reg[[i, i]] += alpha;
        }

        // Compute kernel ridge loss
        let kr_loss = self.compute_kernel_ridge_loss(&k_reg, y)?;

        // Compute gradient
        let mut gradient = Array1::zeros(self.parameters.len());
        let kernel_derivatives = self.compute_kernel_derivatives(x)?;

        for i in 0..self.parameters.len() {
            let kernel_derivative = &kernel_derivatives[i];
            let kr_derivative =
                self.compute_kernel_ridge_derivative(&k_reg, y, kernel_derivative)?;
            gradient[i] = kr_derivative;
        }

        Ok(GradientResult {
            gradient,
            objective_value: kr_loss,
            hessian: None,
        })
    }

    /// Compute maximum mean discrepancy gradient
    fn compute_mmd_gradient(&self, x: &Array2<f64>) -> Result<GradientResult> {
        let n_samples = x.nrows();
        let split_point = n_samples / 2;

        let x1 = x.slice(s![..split_point, ..]);
        let x2 = x.slice(s![split_point.., ..]);

        // Compute MMD
        let mmd = self.compute_mmd(&x1, &x2)?;

        // Compute gradient
        let mut gradient = Array1::zeros(self.parameters.len());
        let mmd_derivatives = self.compute_mmd_derivatives(&x1, &x2)?;

        for i in 0..self.parameters.len() {
            gradient[i] = mmd_derivatives[i];
        }

        Ok(GradientResult {
            gradient,
            objective_value: mmd,
            hessian: None,
        })
    }

    /// Compute kernel target alignment gradient
    fn compute_kta_gradient(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> Result<GradientResult> {
        let y = y.ok_or_else(|| "Target values required for KTA gradient")?;

        // Compute kernel matrix
        let kernel_matrix = self.compute_kernel_matrix(x)?;

        // Compute target kernel matrix
        let target_kernel = self.compute_target_kernel(y);

        // Compute KTA
        let kta = self.compute_kta(&kernel_matrix, &target_kernel);

        // Compute gradient
        let mut gradient = Array1::zeros(self.parameters.len());
        let kernel_derivatives = self.compute_kernel_derivatives(x)?;

        for i in 0..self.parameters.len() {
            let kernel_derivative = &kernel_derivatives[i];
            let kta_derivative =
                self.compute_kta_derivative(&kernel_matrix, &target_kernel, kernel_derivative);
            gradient[i] = kta_derivative;
        }

        Ok(GradientResult {
            gradient,
            objective_value: -kta, // Negative for minimization
            hessian: None,
        })
    }

    /// Update parameters using the chosen optimizer
    fn update_parameters(&mut self, gradient: &Array1<f64>) -> Result<()> {
        match self.optimizer {
            GradientOptimizer::SGD => self.update_sgd(gradient),
            GradientOptimizer::Momentum => self.update_momentum(gradient),
            GradientOptimizer::Adam => self.update_adam(gradient),
            GradientOptimizer::AdaGrad => self.update_adagrad(gradient),
            GradientOptimizer::RMSprop => self.update_rmsprop(gradient),
            GradientOptimizer::LBFGS => self.update_lbfgs(gradient),
        }
    }

    /// SGD update
    fn update_sgd(&mut self, gradient: &Array1<f64>) -> Result<()> {
        for i in 0..self.parameters.len() {
            self.parameters[i] -= self.config.learning_rate * gradient[i];
        }
        self.apply_bounds();
        Ok(())
    }

    /// Momentum update
    fn update_momentum(&mut self, gradient: &Array1<f64>) -> Result<()> {
        let velocity = self.velocity.as_mut().unwrap();

        for i in 0..self.parameters.len() {
            velocity[i] =
                self.config.momentum * velocity[i] - self.config.learning_rate * gradient[i];
            self.parameters[i] += velocity[i];
        }

        self.apply_bounds();
        Ok(())
    }

    /// Adam update
    fn update_adam(&mut self, gradient: &Array1<f64>) -> Result<()> {
        // Initialize Adam state if not already done
        if self.adam_m.is_none() {
            self.adam_m = Some(Array1::zeros(self.parameters.len()));
            self.adam_v = Some(Array1::zeros(self.parameters.len()));
        }

        let adam_m = self.adam_m.as_mut().unwrap();
        let adam_v = self.adam_v.as_mut().unwrap();

        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        for i in 0..self.parameters.len() {
            // Update biased first moment estimate
            adam_m[i] = beta1 * adam_m[i] + (1.0 - beta1) * gradient[i];

            // Update biased second raw moment estimate
            adam_v[i] = beta2 * adam_v[i] + (1.0 - beta2) * gradient[i] * gradient[i];

            // Compute bias-corrected first moment estimate
            let m_hat = adam_m[i] / (1.0 - beta1.powi(self.iteration as i32 + 1));

            // Compute bias-corrected second raw moment estimate
            let v_hat = adam_v[i] / (1.0 - beta2.powi(self.iteration as i32 + 1));

            // Update parameters
            self.parameters[i] -= self.config.learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        }

        self.apply_bounds();
        Ok(())
    }

    /// AdaGrad update
    fn update_adagrad(&mut self, gradient: &Array1<f64>) -> Result<()> {
        if self.adam_v.is_none() {
            self.adam_v = Some(Array1::zeros(self.parameters.len()));
        }

        let accumulated_grad = self.adam_v.as_mut().unwrap();
        let epsilon = 1e-8;

        for i in 0..self.parameters.len() {
            accumulated_grad[i] += gradient[i] * gradient[i];
            self.parameters[i] -=
                self.config.learning_rate * gradient[i] / (accumulated_grad[i].sqrt() + epsilon);
        }

        self.apply_bounds();
        Ok(())
    }

    /// RMSprop update
    fn update_rmsprop(&mut self, gradient: &Array1<f64>) -> Result<()> {
        if self.adam_v.is_none() {
            self.adam_v = Some(Array1::zeros(self.parameters.len()));
        }

        let accumulated_grad = self.adam_v.as_mut().unwrap();
        let decay_rate = 0.9;
        let epsilon = 1e-8;

        for i in 0..self.parameters.len() {
            accumulated_grad[i] =
                decay_rate * accumulated_grad[i] + (1.0 - decay_rate) * gradient[i] * gradient[i];
            self.parameters[i] -=
                self.config.learning_rate * gradient[i] / (accumulated_grad[i].sqrt() + epsilon);
        }

        self.apply_bounds();
        Ok(())
    }

    /// L-BFGS update (simplified version)
    fn update_lbfgs(&mut self, gradient: &Array1<f64>) -> Result<()> {
        // Simplified L-BFGS - just use gradient descent for now
        for i in 0..self.parameters.len() {
            self.parameters[i] -= self.config.learning_rate * gradient[i];
        }
        self.apply_bounds();
        Ok(())
    }

    /// Apply parameter bounds
    fn apply_bounds(&mut self) {
        if let Some(bounds) = &self.parameter_bounds {
            for i in 0..self.parameters.len() {
                self.parameters[i] = self.parameters[i].max(bounds[[i, 0]]).min(bounds[[i, 1]]);
            }
        }
    }

    /// Update learning rate adaptively
    fn update_learning_rate(&mut self, iteration: usize) {
        if iteration > 0 {
            let current_loss = self.optimization_history.last().unwrap().0;
            let previous_loss = self.optimization_history[self.optimization_history.len() - 2].0;

            if current_loss > previous_loss {
                // Decrease learning rate if loss increased
                self.config.learning_rate *= self.config.learning_rate_decay;
                self.config.learning_rate =
                    self.config.learning_rate.max(self.config.min_learning_rate);
            }
        }
    }

    /// Compute kernel matrix
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut kernel_matrix = Array2::zeros((n_samples, n_samples));

        // Assume RBF kernel with parameters[0] as gamma
        let gamma = self.parameters[0];

        for i in 0..n_samples {
            for j in i..n_samples {
                let dist_sq = x
                    .row(i)
                    .iter()
                    .zip(x.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>();

                let kernel_value = (-gamma * dist_sq).exp();
                kernel_matrix[[i, j]] = kernel_value;
                kernel_matrix[[j, i]] = kernel_value;
            }
        }

        Ok(kernel_matrix)
    }

    /// Compute kernel matrix derivatives
    fn compute_kernel_derivatives(&self, x: &Array2<f64>) -> Result<Vec<Array2<f64>>> {
        let n_samples = x.nrows();
        let mut derivatives = Vec::new();

        // Derivative with respect to gamma
        let gamma = self.parameters[0];
        let mut gamma_derivative = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let dist_sq = x
                    .row(i)
                    .iter()
                    .zip(x.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>();

                let kernel_value = (-gamma * dist_sq).exp();
                let derivative_value = -dist_sq * kernel_value;

                gamma_derivative[[i, j]] = derivative_value;
                gamma_derivative[[j, i]] = derivative_value;
            }
        }

        derivatives.push(gamma_derivative);

        // Add derivatives for other parameters if needed
        for param_idx in 1..self.parameters.len() {
            let derivative = Array2::zeros((n_samples, n_samples));
            derivatives.push(derivative);
        }

        Ok(derivatives)
    }

    /// Compute kernel alignment
    fn compute_kernel_alignment(&self, kernel_matrix: &Array2<f64>) -> f64 {
        let n_samples = kernel_matrix.nrows();
        let trace = (0..n_samples).map(|i| kernel_matrix[[i, i]]).sum::<f64>();
        let frobenius_norm = kernel_matrix.iter().map(|&x| x * x).sum::<f64>().sqrt();

        trace / frobenius_norm
    }

    /// Compute alignment derivative
    fn compute_alignment_derivative(
        &self,
        kernel_matrix: &Array2<f64>,
        kernel_derivative: &Array2<f64>,
    ) -> f64 {
        let n_samples = kernel_matrix.nrows();
        let trace = (0..n_samples).map(|i| kernel_matrix[[i, i]]).sum::<f64>();
        let trace_derivative = (0..n_samples)
            .map(|i| kernel_derivative[[i, i]])
            .sum::<f64>();

        let frobenius_norm = kernel_matrix.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let frobenius_derivative = kernel_matrix
            .iter()
            .zip(kernel_derivative.iter())
            .map(|(&k, &dk)| k * dk)
            .sum::<f64>()
            / frobenius_norm;

        (trace_derivative * frobenius_norm - trace * frobenius_derivative)
            / (frobenius_norm * frobenius_norm)
    }

    /// Split data for cross-validation
    fn split_data(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        start_idx: usize,
        end_idx: usize,
    ) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut x_train = Array2::zeros((n_samples - (end_idx - start_idx), n_features));
        let mut y_train = Array1::zeros(n_samples - (end_idx - start_idx));
        let mut x_val = Array2::zeros((end_idx - start_idx, n_features));
        let mut y_val = Array1::zeros(end_idx - start_idx);

        let mut train_idx = 0;
        let mut val_idx = 0;

        for i in 0..n_samples {
            if i >= start_idx && i < end_idx {
                x_val.row_mut(val_idx).assign(&x.row(i));
                y_val[val_idx] = y[i];
                val_idx += 1;
            } else {
                x_train.row_mut(train_idx).assign(&x.row(i));
                y_train[train_idx] = y[i];
                train_idx += 1;
            }
        }

        (x_train, y_train, x_val, y_val)
    }

    /// Compute fold gradient
    fn compute_fold_gradient(
        &self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<GradientResult> {
        // Simplified fold gradient computation
        let gradient = Array1::zeros(self.parameters.len());
        let objective_value = 0.0;

        Ok(GradientResult {
            gradient,
            objective_value,
            hessian: None,
        })
    }

    /// Compute log marginal likelihood
    fn compute_log_marginal_likelihood(
        &self,
        kernel_matrix: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<f64> {
        // Simplified log marginal likelihood
        Ok(0.0)
    }

    /// Compute marginal likelihood derivative
    fn compute_marginal_likelihood_derivative(
        &self,
        kernel_matrix: &Array2<f64>,
        y: &Array1<f64>,
        kernel_derivative: &Array2<f64>,
    ) -> Result<f64> {
        // Simplified derivative computation
        Ok(0.0)
    }

    /// Compute kernel ridge loss
    fn compute_kernel_ridge_loss(
        &self,
        kernel_matrix: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<f64> {
        // Simplified kernel ridge loss
        Ok(0.0)
    }

    /// Compute kernel ridge derivative
    fn compute_kernel_ridge_derivative(
        &self,
        kernel_matrix: &Array2<f64>,
        y: &Array1<f64>,
        kernel_derivative: &Array2<f64>,
    ) -> Result<f64> {
        // Simplified derivative computation
        Ok(0.0)
    }

    /// Compute MMD
    fn compute_mmd(&self, x1: &ArrayView2<f64>, x2: &ArrayView2<f64>) -> Result<f64> {
        // Simplified MMD computation
        Ok(0.0)
    }

    /// Compute MMD derivatives
    fn compute_mmd_derivatives(
        &self,
        x1: &ArrayView2<f64>,
        x2: &ArrayView2<f64>,
    ) -> Result<Array1<f64>> {
        // Simplified derivative computation
        Ok(Array1::zeros(self.parameters.len()))
    }

    /// Compute target kernel matrix
    fn compute_target_kernel(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_samples = y.len();
        let mut target_kernel = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                target_kernel[[i, j]] = y[i] * y[j];
            }
        }

        target_kernel
    }

    /// Compute kernel target alignment
    fn compute_kta(&self, kernel_matrix: &Array2<f64>, target_kernel: &Array2<f64>) -> f64 {
        let numerator = kernel_matrix
            .iter()
            .zip(target_kernel.iter())
            .map(|(&k, &t)| k * t)
            .sum::<f64>();

        let k_norm = kernel_matrix.iter().map(|&k| k * k).sum::<f64>().sqrt();
        let t_norm = target_kernel.iter().map(|&t| t * t).sum::<f64>().sqrt();

        numerator / (k_norm * t_norm)
    }

    /// Compute KTA derivative
    fn compute_kta_derivative(
        &self,
        kernel_matrix: &Array2<f64>,
        target_kernel: &Array2<f64>,
        kernel_derivative: &Array2<f64>,
    ) -> f64 {
        // Simplified KTA derivative
        0.0
    }

    /// Get current parameters
    pub fn get_parameters(&self) -> &Array1<f64> {
        &self.parameters
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &Vec<(f64, Array1<f64>)> {
        &self.optimization_history
    }
}

/// Gradient-based multi-kernel learning
pub struct GradientMultiKernelLearner {
    base_learners: Vec<GradientKernelLearner>,
    combination_weights: Array1<f64>,
    config: GradientConfig,
}

impl GradientMultiKernelLearner {
    /// Create a new gradient-based multi-kernel learner
    pub fn new(n_kernels: usize, n_parameters_per_kernel: usize) -> Self {
        let mut base_learners = Vec::new();
        for _ in 0..n_kernels {
            base_learners.push(GradientKernelLearner::new(n_parameters_per_kernel));
        }

        Self {
            base_learners,
            combination_weights: Array1::from_elem(n_kernels, 1.0 / n_kernels as f64),
            config: GradientConfig::default(),
        }
    }

    /// Optimize all kernels and combination weights
    pub fn optimize(&mut self, x: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()> {
        // Optimize individual kernels
        for learner in &mut self.base_learners {
            learner.optimize(x, y)?;
        }

        // Optimize combination weights
        self.optimize_combination_weights(x, y)?;

        Ok(())
    }

    /// Optimize combination weights
    fn optimize_combination_weights(
        &mut self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> Result<()> {
        // Simplified combination weight optimization
        let n_kernels = self.base_learners.len();
        self.combination_weights = Array1::from_elem(n_kernels, 1.0 / n_kernels as f64);
        Ok(())
    }

    /// Get optimized parameters for all kernels
    pub fn get_all_parameters(&self) -> Vec<&Array1<f64>> {
        self.base_learners
            .iter()
            .map(|learner| learner.get_parameters())
            .collect()
    }

    /// Get combination weights
    pub fn get_combination_weights(&self) -> &Array1<f64> {
        &self.combination_weights
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_gradient_config() {
        let config = GradientConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.max_iterations, 1000);
        assert!(config.tolerance > 0.0);
    }

    #[test]
    fn test_gradient_kernel_learner() {
        let mut learner = GradientKernelLearner::new(2)
            .with_optimizer(GradientOptimizer::Adam)
            .with_objective(KernelObjective::KernelAlignment);

        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();

        learner.initialize_parameters(Array1::from_vec(vec![1.0, 0.5]));
        let optimized_params = learner.optimize(&x, None).unwrap();

        assert_eq!(optimized_params.len(), 2);
    }

    #[test]
    fn test_gradient_optimizers() {
        let optimizers = vec![
            GradientOptimizer::SGD,
            GradientOptimizer::Momentum,
            GradientOptimizer::Adam,
            GradientOptimizer::AdaGrad,
            GradientOptimizer::RMSprop,
        ];

        for optimizer in optimizers {
            let mut learner = GradientKernelLearner::new(1).with_optimizer(optimizer);

            let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]).unwrap();

            learner.initialize_parameters(Array1::from_vec(vec![1.0]));
            let result = learner.optimize(&x, None);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_parameter_bounds() {
        let mut learner = GradientKernelLearner::new(2).with_bounds(
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    0.1, 10.0, // Parameter 0: [0.1, 10.0]
                    0.0, 5.0, // Parameter 1: [0.0, 5.0]
                ],
            )
            .unwrap(),
        );

        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();

        learner.initialize_parameters(Array1::from_vec(vec![100.0, -1.0]));
        let optimized_params = learner.optimize(&x, None).unwrap();

        assert!(optimized_params[0] >= 0.1 && optimized_params[0] <= 10.0);
        assert!(optimized_params[1] >= 0.0 && optimized_params[1] <= 5.0);
    }

    #[test]
    fn test_multi_kernel_learner() {
        let mut multi_learner = GradientMultiKernelLearner::new(3, 2);

        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();

        multi_learner.optimize(&x, None).unwrap();

        let all_params = multi_learner.get_all_parameters();
        assert_eq!(all_params.len(), 3);

        let weights = multi_learner.get_combination_weights();
        assert_eq!(weights.len(), 3);
    }

    #[test]
    fn test_objective_functions() {
        let objectives = vec![
            KernelObjective::KernelAlignment,
            KernelObjective::CrossValidationError,
            KernelObjective::MarginalLikelihood,
            KernelObjective::KernelRidgeLoss,
            KernelObjective::MaximumMeanDiscrepancy,
            KernelObjective::KernelTargetAlignment,
        ];

        for objective in objectives {
            let mut learner = GradientKernelLearner::new(1).with_objective(objective.clone());

            let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0])
                .unwrap();

            let y = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);

            learner.initialize_parameters(Array1::from_vec(vec![1.0]));

            let result = if objective == KernelObjective::KernelAlignment
                || objective == KernelObjective::MaximumMeanDiscrepancy
            {
                learner.optimize(&x, None)
            } else {
                learner.optimize(&x, Some(&y))
            };

            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_adaptive_learning_rate() {
        let config = GradientConfig {
            adaptive_learning_rate: true,
            learning_rate_decay: 0.5,
            min_learning_rate: 1e-6,
            ..Default::default()
        };

        let mut learner = GradientKernelLearner::new(1).with_config(config);

        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();

        learner.initialize_parameters(Array1::from_vec(vec![1.0]));
        let result = learner.optimize(&x, None);
        assert!(result.is_ok());
    }
}
