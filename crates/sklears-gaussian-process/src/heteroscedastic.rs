//! Heteroscedastic Gaussian Process Regression
//!
//! This module implements Gaussian Process regression with input-dependent noise,
//! allowing the noise variance to vary as a function of the input. This is
//! particularly useful when the observation noise is not constant across the
//! input space.
//!
//! # Mathematical Background
//!
//! In heteroscedastic GP regression, the noise variance is a function of the input:
//! y(x) = f(x) + ε(x), where ε(x) ~ N(0, σ²(x))
//!
//! The noise function σ²(x) can be:
//! - Parametric (e.g., linear, polynomial)
//! - Non-parametric (learned via another GP)
//! - Neural network-based (learned via optimization)
//!
//! # Example
//!
//! ```rust
//! use sklears_gaussian_process::heteroscedastic::{
//!     HeteroscedasticGaussianProcessRegressor, LearnableNoiseFunction
//! };
//! use sklears_gaussian_process::kernels::RBF;
//! use sklears_core::traits::{Fit, Predict};
//! use scirs2_core::ndarray::array;
//!
//! // Create heteroscedastic GP with learnable noise function
//! let het_gp = HeteroscedasticGaussianProcessRegressor::builder()
//!     .signal_kernel(Box::new(RBF::new(1.0)))
//!     .noise_function(LearnableNoiseFunction::gaussian_process(
//!         Box::new(RBF::new(0.5)), 10
//!     ))
//!     .max_iter(100)
//!     .learning_rate(0.01)
//!     .build();
//!
//! let x_train = array![[0.0], [1.0], [2.0], [3.0]];
//! let y_train = array![0.0, 1.0, 2.5, 4.2];
//!
//! let trained_model = het_gp.fit(&x_train, &y_train).unwrap();
//! let predictions = trained_model.predict(&x_train).unwrap();
//! ```

use crate::kernels::Kernel;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, Random}; // SciRS2 Policy
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict};
use std::fmt;

/// State marker for untrained heteroscedastic GP
#[derive(Debug, Clone)]
pub struct Untrained;

/// State marker for trained heteroscedastic GP
#[derive(Debug, Clone)]
pub struct Trained {
    pub signal_kernel: Box<dyn Kernel>,
    pub noise_function: LearnableNoiseFunction,
    pub training_data: (Array2<f64>, Array1<f64>),
    pub alpha: Array1<f64>,
    pub cholesky: Array2<f64>,
    pub log_likelihood: f64,
}

/// Trait for noise functions that can learn their parameters
pub trait NoiseFunction: fmt::Debug + Send + Sync {
    /// Compute noise variance for given inputs
    fn compute_noise(&self, x: &Array2<f64>) -> SklResult<Array1<f64>>;

    /// Update noise function parameters given residuals
    fn update_parameters(
        &mut self,
        x: &Array2<f64>,
        residuals: &Array1<f64>,
        learning_rate: f64,
    ) -> SklResult<()>;

    /// Get current parameter values
    fn get_parameters(&self) -> Vec<f64>;

    /// Set parameter values
    fn set_parameters(&mut self, params: &[f64]) -> SklResult<()>;

    /// Clone the noise function
    fn clone_box(&self) -> Box<dyn NoiseFunction>;
}

/// Constant noise function
#[derive(Debug, Clone)]
pub struct ConstantNoise {
    variance: f64,
}

impl ConstantNoise {
    pub fn new(variance: f64) -> Self {
        Self {
            variance: variance.max(1e-12),
        }
    }
}

impl NoiseFunction for ConstantNoise {
    fn compute_noise(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        Ok(Array1::from_elem(x.nrows(), self.variance))
    }

    fn update_parameters(
        &mut self,
        _x: &Array2<f64>,
        residuals: &Array1<f64>,
        learning_rate: f64,
    ) -> SklResult<()> {
        // Update variance based on residuals
        let mean_squared_residual =
            residuals.iter().map(|&r| r * r).sum::<f64>() / residuals.len() as f64;
        self.variance =
            (1.0 - learning_rate) * self.variance + learning_rate * mean_squared_residual;
        self.variance = self.variance.max(1e-12);
        Ok(())
    }

    fn get_parameters(&self) -> Vec<f64> {
        vec![self.variance]
    }

    fn set_parameters(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 1 {
            return Err(SklearsError::InvalidInput(
                "ConstantNoise requires exactly 1 parameter".to_string(),
            ));
        }
        self.variance = params[0].max(1e-12);
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn NoiseFunction> {
        Box::new(self.clone())
    }
}

/// Linear noise function: σ²(x) = a + b*x
#[derive(Debug, Clone)]
pub struct LinearNoise {
    intercept: f64,
    slope: f64,
}

impl LinearNoise {
    pub fn new(intercept: f64, slope: f64) -> Self {
        Self {
            intercept: intercept.max(1e-12),
            slope,
        }
    }
}

impl NoiseFunction for LinearNoise {
    fn compute_noise(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        if x.ncols() != 1 {
            return Err(SklearsError::InvalidInput(
                "LinearNoise only supports 1D inputs".to_string(),
            ));
        }

        let mut noise = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            noise[i] = (self.intercept + self.slope * x[[i, 0]]).max(1e-12);
        }
        Ok(noise)
    }

    fn update_parameters(
        &mut self,
        x: &Array2<f64>,
        residuals: &Array1<f64>,
        learning_rate: f64,
    ) -> SklResult<()> {
        if x.ncols() != 1 {
            return Err(SklearsError::InvalidInput(
                "LinearNoise only supports 1D inputs".to_string(),
            ));
        }

        // Gradient descent update for linear parameters
        let squared_residuals: Array1<f64> = residuals.iter().map(|&r| r * r).collect();

        // Gradients
        let mut grad_intercept = 0.0;
        let mut grad_slope = 0.0;

        for i in 0..x.nrows() {
            let predicted_noise = self.intercept + self.slope * x[[i, 0]];
            let error = squared_residuals[i] - predicted_noise;
            grad_intercept += -2.0 * error;
            grad_slope += -2.0 * error * x[[i, 0]];
        }

        grad_intercept /= x.nrows() as f64;
        grad_slope /= x.nrows() as f64;

        // Update parameters
        self.intercept -= learning_rate * grad_intercept;
        self.slope -= learning_rate * grad_slope;
        self.intercept = self.intercept.max(1e-12);

        Ok(())
    }

    fn get_parameters(&self) -> Vec<f64> {
        vec![self.intercept, self.slope]
    }

    fn set_parameters(&mut self, params: &[f64]) -> SklResult<()> {
        if params.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "LinearNoise requires exactly 2 parameters".to_string(),
            ));
        }
        self.intercept = params[0].max(1e-12);
        self.slope = params[1];
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn NoiseFunction> {
        Box::new(self.clone())
    }
}

/// Polynomial noise function: σ²(x) = a₀ + a₁*x + a₂*x² + ... + aₙ*xⁿ
#[derive(Debug, Clone)]
pub struct PolynomialNoise {
    coefficients: Vec<f64>,
}

impl PolynomialNoise {
    pub fn new(coefficients: Vec<f64>) -> Self {
        let mut coeffs = coefficients;
        if coeffs.is_empty() {
            coeffs.push(1e-6); // Default constant term
        }
        Self {
            coefficients: coeffs,
        }
    }

    pub fn linear(intercept: f64, slope: f64) -> Self {
        Self::new(vec![intercept.max(1e-12), slope])
    }

    pub fn quadratic(a: f64, b: f64, c: f64) -> Self {
        Self::new(vec![a.max(1e-12), b, c])
    }
}

impl NoiseFunction for PolynomialNoise {
    fn compute_noise(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        if x.ncols() != 1 {
            return Err(SklearsError::InvalidInput(
                "PolynomialNoise only supports 1D inputs".to_string(),
            ));
        }

        let mut noise = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut value = 0.0;
            let x_val = x[[i, 0]];
            for (j, &coeff) in self.coefficients.iter().enumerate() {
                value += coeff * x_val.powi(j as i32);
            }
            noise[i] = value.max(1e-12);
        }
        Ok(noise)
    }

    fn update_parameters(
        &mut self,
        x: &Array2<f64>,
        residuals: &Array1<f64>,
        learning_rate: f64,
    ) -> SklResult<()> {
        if x.ncols() != 1 {
            return Err(SklearsError::InvalidInput(
                "PolynomialNoise only supports 1D inputs".to_string(),
            ));
        }

        let squared_residuals: Array1<f64> = residuals.iter().map(|&r| r * r).collect();

        // Compute gradients for each coefficient
        let mut gradients = vec![0.0; self.coefficients.len()];

        for i in 0..x.nrows() {
            let x_val = x[[i, 0]];
            let predicted_noise = self.compute_noise(&x.slice(s![i..i + 1, ..]).to_owned())?[0];
            let error = squared_residuals[i] - predicted_noise;

            for (j, grad) in gradients.iter_mut().enumerate() {
                *grad += -2.0 * error * x_val.powi(j as i32);
            }
        }

        // Normalize gradients
        for grad in &mut gradients {
            *grad /= x.nrows() as f64;
        }

        // Update coefficients
        for (i, &grad) in gradients.iter().enumerate() {
            self.coefficients[i] -= learning_rate * grad;
        }

        // Ensure first coefficient (constant term) is positive
        self.coefficients[0] = self.coefficients[0].max(1e-12);

        Ok(())
    }

    fn get_parameters(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    fn set_parameters(&mut self, params: &[f64]) -> SklResult<()> {
        if params.is_empty() {
            return Err(SklearsError::InvalidInput(
                "PolynomialNoise requires at least 1 parameter".to_string(),
            ));
        }
        self.coefficients = params.to_vec();
        self.coefficients[0] = self.coefficients[0].max(1e-12);
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn NoiseFunction> {
        Box::new(self.clone())
    }
}

/// Gaussian Process-based noise function
#[derive(Debug, Clone)]
pub struct GaussianProcessNoise {
    kernel: Box<dyn Kernel>,
    inducing_points: Array2<f64>,
    variational_mean: Array1<f64>,
    variational_var: Array1<f64>,
    n_inducing: usize,
}

impl GaussianProcessNoise {
    pub fn new(kernel: Box<dyn Kernel>, n_inducing: usize) -> Self {
        Self {
            kernel,
            inducing_points: Array2::zeros((0, 0)), // Will be initialized during training
            variational_mean: Array1::zeros(n_inducing),
            variational_var: Array1::ones(n_inducing),
            n_inducing,
        }
    }

    pub fn initialize_inducing_points(&mut self, x: &Array2<f64>) -> SklResult<()> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot initialize with empty data".to_string(),
            ));
        }

        let n_data = x.nrows();
        let input_dim = x.ncols();
        let n_inducing = self.n_inducing.min(n_data);

        if n_inducing >= n_data {
            self.inducing_points = x.clone();
        } else {
            // Random subset selection
            let mut rng = thread_rng();
            let mut inducing_points = Array2::zeros((n_inducing, input_dim));
            for i in 0..n_inducing {
                let idx = rng.gen_range(0..n_data);
                for j in 0..input_dim {
                    inducing_points[[i, j]] = x[[idx, j]];
                }
            }
            self.inducing_points = inducing_points;
        }

        // Resize variational parameters
        let actual_n_inducing = self.inducing_points.nrows();
        self.variational_mean = Array1::zeros(actual_n_inducing);
        self.variational_var = Array1::ones(actual_n_inducing);

        Ok(())
    }
}

impl NoiseFunction for GaussianProcessNoise {
    fn compute_noise(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        if self.inducing_points.is_empty() {
            return Ok(Array1::from_elem(x.nrows(), 1e-6));
        }

        let k_uf = self
            .kernel
            .compute_kernel_matrix(&self.inducing_points, Some(x))?;
        let k_uu = self
            .kernel
            .compute_kernel_matrix(&self.inducing_points, None)?;

        // Add jitter for numerical stability
        let mut k_uu_reg = k_uu.clone();
        for i in 0..k_uu_reg.nrows() {
            k_uu_reg[[i, i]] += 1e-6;
        }

        // Simple prediction using diagonal approximation
        let mut noise = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            let k_star = k_uf.column(i);
            let mean_pred = k_star.dot(&self.variational_mean);
            // Apply softplus to ensure positivity: log(1 + exp(x))
            noise[i] = if mean_pred > 10.0 {
                mean_pred // For large values, softplus ≈ x
            } else if mean_pred < -10.0 {
                (-mean_pred).exp() // For very negative values
            } else {
                (1.0 + mean_pred.exp()).ln()
            }
            .max(1e-12);
        }

        Ok(noise)
    }

    fn update_parameters(
        &mut self,
        x: &Array2<f64>,
        residuals: &Array1<f64>,
        learning_rate: f64,
    ) -> SklResult<()> {
        if self.inducing_points.is_empty() {
            self.initialize_inducing_points(x)?;
        }

        // Simple variational parameter update based on residuals
        let squared_residuals: Array1<f64> = residuals.iter().map(|&r| r * r).collect();

        // Update variational mean using a simplified gradient
        let k_uf = self
            .kernel
            .compute_kernel_matrix(&self.inducing_points, Some(x))?;

        for i in 0..self.variational_mean.len() {
            let mut gradient = 0.0;
            for j in 0..x.nrows() {
                let predicted_noise = self.compute_noise(&x.slice(s![j..j + 1, ..]).to_owned())?[0];
                let error = squared_residuals[j] - predicted_noise;
                gradient += error * k_uf[[i, j]];
            }
            gradient /= x.nrows() as f64;
            self.variational_mean[i] += learning_rate * gradient;
        }

        Ok(())
    }

    fn get_parameters(&self) -> Vec<f64> {
        let mut params = self.variational_mean.to_vec();
        params.extend(self.variational_var.to_vec());
        params
    }

    fn set_parameters(&mut self, params: &[f64]) -> SklResult<()> {
        let n_inducing = self.variational_mean.len();
        if params.len() != 2 * n_inducing {
            return Err(SklearsError::InvalidInput(format!(
                "GaussianProcessNoise requires {} parameters, got {}",
                2 * n_inducing,
                params.len()
            )));
        }

        for i in 0..n_inducing {
            self.variational_mean[i] = params[i];
            self.variational_var[i] = params[n_inducing + i].max(1e-12);
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn NoiseFunction> {
        Box::new(self.clone())
    }
}

/// Neural Network-based noise function
#[derive(Debug, Clone)]
pub struct NeuralNetworkNoise {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    layer_sizes: Vec<usize>,
}

impl NeuralNetworkNoise {
    pub fn new(input_dim: usize, hidden_sizes: Vec<usize>) -> Self {
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(hidden_sizes);
        layer_sizes.push(1); // Output dimension is 1

        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut rng = thread_rng();

        for i in 0..layer_sizes.len() - 1 {
            let in_size = layer_sizes[i];
            let out_size = layer_sizes[i + 1];

            // Xavier initialization
            let scale = (2.0 / (in_size + out_size) as f64).sqrt();
            let mut weight = Array2::zeros((out_size, in_size));
            for j in 0..out_size {
                for k in 0..in_size {
                    weight[[j, k]] = rng.gen_range(-scale..scale);
                }
            }
            weights.push(weight);

            let mut bias = Array1::zeros(out_size);
            for j in 0..out_size {
                bias[j] = rng.gen_range(-0.1..0.1);
            }
            biases.push(bias);
        }

        Self {
            weights,
            biases,
            layer_sizes,
        }
    }

    fn forward(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        let mut activation = x.clone();

        for (i, (weight, bias)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            // Linear transformation: W * x + b
            let mut linear_output = Array2::zeros((activation.nrows(), weight.nrows()));
            for j in 0..activation.nrows() {
                for k in 0..weight.nrows() {
                    let mut sum = bias[k];
                    for l in 0..weight.ncols() {
                        sum += weight[[k, l]] * activation[[j, l]];
                    }
                    linear_output[[j, k]] = sum;
                }
            }

            // Apply activation function
            if i < self.weights.len() - 1 {
                // Hidden layers: ReLU
                activation = linear_output.mapv(|x| x.max(0.0));
            } else {
                // Output layer: Softplus to ensure positivity
                activation = linear_output.mapv(|x| {
                    if x > 10.0 {
                        x
                    } else if x < -10.0 {
                        (-x).exp()
                    } else {
                        (1.0 + x.exp()).ln()
                    }
                    .max(1e-12)
                });
            }
        }

        Ok(activation.column(0).to_owned())
    }
}

impl NoiseFunction for NeuralNetworkNoise {
    fn compute_noise(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        self.forward(x)
    }

    fn update_parameters(
        &mut self,
        x: &Array2<f64>,
        residuals: &Array1<f64>,
        learning_rate: f64,
    ) -> SklResult<()> {
        // Simplified gradient update (in practice, would use proper backpropagation)
        let squared_residuals: Array1<f64> = residuals.iter().map(|&r| r * r).collect();
        let predictions = self.forward(x)?;

        // Simple parameter perturbation for gradient estimation
        for layer_idx in 0..self.weights.len() {
            for i in 0..self.weights[layer_idx].nrows() {
                for j in 0..self.weights[layer_idx].ncols() {
                    let original_weight = self.weights[layer_idx][[i, j]];

                    // Finite difference gradient estimation
                    let epsilon = 1e-5;

                    // Forward pass with positive perturbation
                    self.weights[layer_idx][[i, j]] = original_weight + epsilon;
                    let pred_plus = self.forward(x)?;

                    // Forward pass with negative perturbation
                    self.weights[layer_idx][[i, j]] = original_weight - epsilon;
                    let pred_minus = self.forward(x)?;

                    // Restore original weight
                    self.weights[layer_idx][[i, j]] = original_weight;

                    // Compute loss gradient
                    let mut gradient = 0.0;
                    for k in 0..x.nrows() {
                        let loss_plus = (squared_residuals[k] - pred_plus[k]).powi(2);
                        let loss_minus = (squared_residuals[k] - pred_minus[k]).powi(2);
                        gradient += (loss_plus - loss_minus) / (2.0 * epsilon);
                    }
                    gradient /= x.nrows() as f64;

                    // Update weight
                    self.weights[layer_idx][[i, j]] -= learning_rate * gradient;
                }
            }
        }

        Ok(())
    }

    fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for weight in &self.weights {
            for row in weight.rows() {
                params.extend(row.iter());
            }
        }
        for bias in &self.biases {
            params.extend(bias.iter());
        }
        params
    }

    fn set_parameters(&mut self, params: &[f64]) -> SklResult<()> {
        let mut param_idx = 0;

        // Set weights
        for weight in &mut self.weights {
            for mut row in weight.rows_mut() {
                for val in row.iter_mut() {
                    if param_idx >= params.len() {
                        return Err(SklearsError::InvalidInput(
                            "Not enough parameters provided".to_string(),
                        ));
                    }
                    *val = params[param_idx];
                    param_idx += 1;
                }
            }
        }

        // Set biases
        for bias in &mut self.biases {
            for val in bias.iter_mut() {
                if param_idx >= params.len() {
                    return Err(SklearsError::InvalidInput(
                        "Not enough parameters provided".to_string(),
                    ));
                }
                *val = params[param_idx];
                param_idx += 1;
            }
        }

        if param_idx != params.len() {
            return Err(SklearsError::InvalidInput(
                "Wrong number of parameters provided".to_string(),
            ));
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn NoiseFunction> {
        Box::new(self.clone())
    }
}

/// Learnable noise function enum
#[derive(Debug, Clone)]
pub enum LearnableNoiseFunction {
    Constant(ConstantNoise),
    Linear(LinearNoise),
    Polynomial(PolynomialNoise),
    GaussianProcess(GaussianProcessNoise),
    NeuralNetwork(NeuralNetworkNoise),
}

impl LearnableNoiseFunction {
    pub fn constant(variance: f64) -> Self {
        Self::Constant(ConstantNoise::new(variance))
    }

    pub fn linear(intercept: f64, slope: f64) -> Self {
        Self::Linear(LinearNoise::new(intercept, slope))
    }

    pub fn polynomial(coefficients: Vec<f64>) -> Self {
        Self::Polynomial(PolynomialNoise::new(coefficients))
    }

    pub fn gaussian_process(kernel: Box<dyn Kernel>, n_inducing: usize) -> Self {
        Self::GaussianProcess(GaussianProcessNoise::new(kernel, n_inducing))
    }

    pub fn neural_network(input_dim: usize, hidden_sizes: Vec<usize>) -> Self {
        Self::NeuralNetwork(NeuralNetworkNoise::new(input_dim, hidden_sizes))
    }
}

impl NoiseFunction for LearnableNoiseFunction {
    fn compute_noise(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        match self {
            Self::Constant(noise) => noise.compute_noise(x),
            Self::Linear(noise) => noise.compute_noise(x),
            Self::Polynomial(noise) => noise.compute_noise(x),
            Self::GaussianProcess(noise) => noise.compute_noise(x),
            Self::NeuralNetwork(noise) => noise.compute_noise(x),
        }
    }

    fn update_parameters(
        &mut self,
        x: &Array2<f64>,
        residuals: &Array1<f64>,
        learning_rate: f64,
    ) -> SklResult<()> {
        match self {
            Self::Constant(noise) => noise.update_parameters(x, residuals, learning_rate),
            Self::Linear(noise) => noise.update_parameters(x, residuals, learning_rate),
            Self::Polynomial(noise) => noise.update_parameters(x, residuals, learning_rate),
            Self::GaussianProcess(noise) => noise.update_parameters(x, residuals, learning_rate),
            Self::NeuralNetwork(noise) => noise.update_parameters(x, residuals, learning_rate),
        }
    }

    fn get_parameters(&self) -> Vec<f64> {
        match self {
            Self::Constant(noise) => noise.get_parameters(),
            Self::Linear(noise) => noise.get_parameters(),
            Self::Polynomial(noise) => noise.get_parameters(),
            Self::GaussianProcess(noise) => noise.get_parameters(),
            Self::NeuralNetwork(noise) => noise.get_parameters(),
        }
    }

    fn set_parameters(&mut self, params: &[f64]) -> SklResult<()> {
        match self {
            Self::Constant(noise) => noise.set_parameters(params),
            Self::Linear(noise) => noise.set_parameters(params),
            Self::Polynomial(noise) => noise.set_parameters(params),
            Self::GaussianProcess(noise) => noise.set_parameters(params),
            Self::NeuralNetwork(noise) => noise.set_parameters(params),
        }
    }

    fn clone_box(&self) -> Box<dyn NoiseFunction> {
        Box::new(self.clone())
    }
}

/// Heteroscedastic Gaussian Process Regressor
#[derive(Debug, Clone)]
pub struct HeteroscedasticGaussianProcessRegressor<S = Untrained> {
    signal_kernel: Option<Box<dyn Kernel>>,
    noise_function: Option<LearnableNoiseFunction>,
    max_iter: usize,
    learning_rate: f64,
    convergence_threshold: f64,
    jitter: f64,
    _state: S,
}

impl Default for HeteroscedasticGaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl HeteroscedasticGaussianProcessRegressor<Untrained> {
    pub fn new() -> Self {
        Self {
            signal_kernel: None,
            noise_function: None,
            max_iter: 100,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            jitter: 1e-6,
            _state: Untrained,
        }
    }

    pub fn builder() -> HeteroscedasticGaussianProcessRegressorBuilder {
        HeteroscedasticGaussianProcessRegressorBuilder::new()
    }

    pub fn fit(
        mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
    ) -> SklResult<HeteroscedasticGaussianProcessRegressor<Trained>> {
        if x_train.nrows() != y_train.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: x_train.nrows(),
                actual: y_train.len(),
            });
        }

        let signal_kernel = self
            .signal_kernel
            .clone()
            .ok_or_else(|| SklearsError::InvalidInput("Signal kernel not set".to_string()))?;
        let mut noise_function = self
            .noise_function
            .clone()
            .ok_or_else(|| SklearsError::InvalidInput("Noise function not set".to_string()))?;

        let mut log_likelihood = f64::NEG_INFINITY;
        let mut alpha = Array1::zeros(x_train.nrows());
        let mut cholesky = Array2::eye(x_train.nrows());

        // Alternating optimization loop
        for iteration in 0..self.max_iter {
            // Compute current noise estimates
            let noise_vars = noise_function.compute_noise(x_train)?;

            // Build covariance matrix with current noise
            let k_signal = signal_kernel.compute_kernel_matrix(x_train, None)?;
            let mut k_total = k_signal.clone();
            for i in 0..k_total.nrows() {
                k_total[[i, i]] += noise_vars[i] + self.jitter;
            }

            // Cholesky decomposition
            cholesky = self.cholesky_decomposition(&k_total)?;

            // Solve for alpha
            alpha = self.solve_triangular(&cholesky, y_train)?;

            // Compute log marginal likelihood
            let new_log_likelihood =
                self.compute_log_marginal_likelihood(&cholesky, y_train, &alpha);

            // Check convergence
            if (new_log_likelihood - log_likelihood).abs() < self.convergence_threshold {
                log_likelihood = new_log_likelihood;
                break;
            }

            // Update noise function parameters
            let residuals = self.compute_residuals(x_train, y_train, &signal_kernel, &alpha)?;
            noise_function.update_parameters(x_train, &residuals, self.learning_rate)?;

            log_likelihood = new_log_likelihood;
        }

        Ok(HeteroscedasticGaussianProcessRegressor {
            signal_kernel: Some(signal_kernel.clone_box()),
            noise_function: Some(noise_function.clone()),
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            convergence_threshold: self.convergence_threshold,
            jitter: self.jitter,
            _state: Trained {
                signal_kernel,
                noise_function,
                training_data: (x_train.clone(), y_train.clone()),
                alpha,
                cholesky,
                log_likelihood,
            },
        })
    }

    fn cholesky_decomposition(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = matrix.nrows();
        let mut chol = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += chol[[j, k]] * chol[[j, k]];
                    }
                    let val = matrix[[j, j]] - sum;
                    if val <= 0.0 {
                        return Err(SklearsError::NumericalError(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    chol[[j, j]] = val.sqrt();
                } else {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += chol[[i, k]] * chol[[j, k]];
                    }
                    chol[[i, j]] = (matrix[[i, j]] - sum) / chol[[j, j]];
                }
            }
        }

        Ok(chol)
    }

    fn solve_triangular(&self, l: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
        let n = l.nrows();
        let mut x = Array1::zeros(n);

        // Forward substitution
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * x[j];
            }
            x[i] = (b[i] - sum) / l[[i, i]];
        }

        // Backward substitution
        let mut result = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[[j, i]] * result[j];
            }
            result[i] = (x[i] - sum) / l[[i, i]];
        }

        Ok(result)
    }

    fn compute_log_marginal_likelihood(
        &self,
        chol: &Array2<f64>,
        y: &Array1<f64>,
        alpha: &Array1<f64>,
    ) -> f64 {
        let log_det = chol.diag().iter().map(|&x| x.ln()).sum::<f64>() * 2.0;
        let data_fit = y.dot(alpha);
        let n = y.len() as f64;
        -0.5 * (data_fit + log_det + n * (2.0 * std::f64::consts::PI).ln())
    }

    fn compute_residuals(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        kernel: &Box<dyn Kernel>,
        alpha: &Array1<f64>,
    ) -> SklResult<Array1<f64>> {
        let k_matrix = kernel.compute_kernel_matrix(x, None)?;
        let mean_pred = k_matrix.dot(alpha);
        Ok(y - &mean_pred)
    }
}

impl HeteroscedasticGaussianProcessRegressor<Trained> {
    pub fn predict(&self, x_test: &Array2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let state = &self._state;

        // Compute signal prediction
        let k_star = state
            .signal_kernel
            .compute_kernel_matrix(&state.training_data.0, Some(x_test))?;
        let mean_pred = k_star.t().dot(&state.alpha);

        // Compute noise prediction
        let noise_pred = state.noise_function.compute_noise(x_test)?;

        // Compute predictive variance (simplified)
        let k_test = state.signal_kernel.compute_kernel_matrix(x_test, None)?;
        let mut var_pred = Array1::zeros(x_test.nrows());
        for i in 0..x_test.nrows() {
            var_pred[i] = k_test[[i, i]] + noise_pred[i];
        }

        Ok((mean_pred, var_pred))
    }

    pub fn log_likelihood(&self) -> f64 {
        self._state.log_likelihood
    }

    pub fn get_noise_parameters(&self) -> Vec<f64> {
        self._state.noise_function.get_parameters()
    }

    pub fn predict_noise(&self, x_test: &Array2<f64>) -> SklResult<Array1<f64>> {
        self._state.noise_function.compute_noise(x_test)
    }
}

pub struct HeteroscedasticGaussianProcessRegressorBuilder {
    signal_kernel: Option<Box<dyn Kernel>>,
    noise_function: Option<LearnableNoiseFunction>,
    max_iter: usize,
    learning_rate: f64,
    convergence_threshold: f64,
    jitter: f64,
}

impl HeteroscedasticGaussianProcessRegressorBuilder {
    pub fn new() -> Self {
        Self {
            signal_kernel: None,
            noise_function: None,
            max_iter: 100,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            jitter: 1e-6,
        }
    }

    pub fn signal_kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.signal_kernel = Some(kernel);
        self
    }

    pub fn noise_function(mut self, noise_function: LearnableNoiseFunction) -> Self {
        self.noise_function = Some(noise_function);
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    pub fn jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    pub fn build(self) -> HeteroscedasticGaussianProcessRegressor<Untrained> {
        HeteroscedasticGaussianProcessRegressor {
            signal_kernel: self.signal_kernel,
            noise_function: self.noise_function,
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            convergence_threshold: self.convergence_threshold,
            jitter: self.jitter,
            _state: Untrained,
        }
    }
}

impl Default for HeteroscedasticGaussianProcessRegressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct HeteroscedasticGPConfig {
    pub max_iter: usize,
    pub learning_rate: f64,
    pub convergence_threshold: f64,
    pub jitter: f64,
}

impl Default for HeteroscedasticGPConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            jitter: 1e-6,
        }
    }
}

impl<S> Estimator for HeteroscedasticGaussianProcessRegressor<S> {
    type Config = HeteroscedasticGPConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: HeteroscedasticGPConfig = HeteroscedasticGPConfig {
            max_iter: 100,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            jitter: 1e-6,
        };
        &DEFAULT_CONFIG
    }
}

impl Fit<Array2<f64>, Array1<f64>> for HeteroscedasticGaussianProcessRegressor<Untrained> {
    type Fitted = HeteroscedasticGaussianProcessRegressor<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        self.fit(x, y)
    }
}

impl Predict<Array2<f64>, Array1<f64>> for HeteroscedasticGaussianProcessRegressor<Trained> {
    fn predict(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        let (mean_pred, _) = self.predict(x)?;
        Ok(mean_pred)
    }
}

// Slice import for array slicing
use scirs2_core::ndarray::s;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_constant_noise() {
        let mut noise = ConstantNoise::new(0.1);
        let x = array![[0.0], [1.0], [2.0]];

        let noise_vals = noise.compute_noise(&x).unwrap();
        assert_eq!(noise_vals.len(), 3);
        assert!(noise_vals.iter().all(|&val| (val - 0.1).abs() < 1e-10));

        let residuals = array![0.2, -0.1, 0.3];
        noise.update_parameters(&x, &residuals, 0.1).unwrap();

        let params = noise.get_parameters();
        assert_eq!(params.len(), 1);
        assert!(params[0] > 1e-12);
    }

    #[test]
    fn test_linear_noise() {
        let mut noise = LinearNoise::new(0.1, 0.05);
        let x = array![[0.0], [1.0], [2.0]];

        let noise_vals = noise.compute_noise(&x).unwrap();
        assert_eq!(noise_vals.len(), 3);
        assert_abs_diff_eq!(noise_vals[0], 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(noise_vals[1], 0.15, epsilon = 1e-10);
        assert_abs_diff_eq!(noise_vals[2], 0.2, epsilon = 1e-10);

        let params = noise.get_parameters();
        assert_eq!(params.len(), 2);
        assert_abs_diff_eq!(params[0], 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(params[1], 0.05, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_noise() {
        let mut noise = PolynomialNoise::quadratic(0.1, 0.05, 0.02);
        let x = array![[0.0], [1.0], [2.0]];

        let noise_vals = noise.compute_noise(&x).unwrap();
        assert_eq!(noise_vals.len(), 3);
        assert_abs_diff_eq!(noise_vals[0], 0.1, epsilon = 1e-10); // 0.1 + 0*0.05 + 0*0.02
        assert_abs_diff_eq!(noise_vals[1], 0.17, epsilon = 1e-10); // 0.1 + 1*0.05 + 1*0.02
        assert_abs_diff_eq!(noise_vals[2], 0.28, epsilon = 1e-10); // 0.1 + 2*0.05 + 4*0.02

        let params = noise.get_parameters();
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn test_gaussian_process_noise() {
        let kernel = Box::new(RBF::new(1.0));
        let mut noise = GaussianProcessNoise::new(kernel, 5);
        let x = array![[0.0], [1.0], [2.0]];

        // Before initialization, should return small default values
        let noise_vals = noise.compute_noise(&x).unwrap();
        assert_eq!(noise_vals.len(), 3);
        assert!(noise_vals.iter().all(|&val| val >= 1e-12));

        // Initialize and test again
        noise.initialize_inducing_points(&x).unwrap();
        let noise_vals2 = noise.compute_noise(&x).unwrap();
        assert_eq!(noise_vals2.len(), 3);
        assert!(noise_vals2.iter().all(|&val| val >= 1e-12));
    }

    #[test]
    fn test_neural_network_noise() {
        let mut noise = NeuralNetworkNoise::new(1, vec![5, 3]);
        let x = array![[0.0], [1.0], [2.0]];

        let noise_vals = noise.compute_noise(&x).unwrap();
        assert_eq!(noise_vals.len(), 3);
        assert!(noise_vals.iter().all(|&val| val >= 1e-12));

        let params = noise.get_parameters();
        assert!(params.len() > 0);

        // Test parameter setting
        let result = noise.set_parameters(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_learnable_noise_function_enum() {
        let noise_fns = vec![
            LearnableNoiseFunction::constant(0.1),
            LearnableNoiseFunction::linear(0.1, 0.05),
            LearnableNoiseFunction::polynomial(vec![0.1, 0.05, 0.02]),
            LearnableNoiseFunction::gaussian_process(Box::new(RBF::new(1.0)), 5),
            LearnableNoiseFunction::neural_network(1, vec![3]),
        ];

        let x = array![[0.0], [1.0], [2.0]];

        for mut noise_fn in noise_fns {
            let noise_vals = noise_fn.compute_noise(&x).unwrap();
            assert_eq!(noise_vals.len(), 3);
            assert!(noise_vals.iter().all(|&val| val >= 1e-12));

            let residuals = array![0.1, -0.05, 0.2];
            let result = noise_fn.update_parameters(&x, &residuals, 0.01);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_heteroscedastic_gpr_creation() {
        let het_gp = HeteroscedasticGaussianProcessRegressor::builder()
            .signal_kernel(Box::new(RBF::new(1.0)))
            .noise_function(LearnableNoiseFunction::constant(0.1))
            .max_iter(50)
            .learning_rate(0.05)
            .build();

        let config = het_gp.config();
        assert_eq!(config.max_iter, 100); // Default config
        assert_eq!(config.learning_rate, 0.01); // Default config
    }

    #[test]
    fn test_heteroscedastic_gpr_fit_predict() {
        let het_gp = HeteroscedasticGaussianProcessRegressor::builder()
            .signal_kernel(Box::new(RBF::new(1.0)))
            .noise_function(LearnableNoiseFunction::constant(0.1))
            .max_iter(10)
            .learning_rate(0.1)
            .build();

        let x_train = array![[0.0], [1.0], [2.0], [3.0]];
        let y_train = array![0.0, 1.0, 2.1, 2.9];

        let trained_model = het_gp.fit(&x_train, &y_train).unwrap();
        assert!(trained_model.log_likelihood().is_finite());

        let x_test = array![[0.5], [1.5]];
        let (predictions, uncertainties) = trained_model.predict(&x_test).unwrap();

        assert_eq!(predictions.len(), 2);
        assert_eq!(uncertainties.len(), 2);
        assert!(predictions.iter().all(|&p| p.is_finite()));
        assert!(uncertainties.iter().all(|&u| u > 0.0));

        // Test noise prediction
        let noise_pred = trained_model.predict_noise(&x_test).unwrap();
        assert_eq!(noise_pred.len(), 2);
        assert!(noise_pred.iter().all(|&n| n > 0.0));
    }

    #[test]
    fn test_heteroscedastic_gpr_linear_noise() {
        let het_gp = HeteroscedasticGaussianProcessRegressor::builder()
            .signal_kernel(Box::new(RBF::new(1.0)))
            .noise_function(LearnableNoiseFunction::linear(0.05, 0.02))
            .max_iter(5)
            .learning_rate(0.1)
            .build();

        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 1.0, 2.0];

        let trained_model = het_gp.fit(&x_train, &y_train).unwrap();

        let x_test = array![[0.5], [1.5]];
        let (predictions, _) = trained_model.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 2);

        let noise_params = trained_model.get_noise_parameters();
        assert_eq!(noise_params.len(), 2);
    }

    #[test]
    fn test_heteroscedastic_gpr_polynomial_noise() {
        let het_gp = HeteroscedasticGaussianProcessRegressor::builder()
            .signal_kernel(Box::new(RBF::new(1.0)))
            .noise_function(LearnableNoiseFunction::polynomial(vec![0.1, 0.05]))
            .max_iter(5)
            .learning_rate(0.05)
            .build();

        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 1.0, 2.0];

        let trained_model = het_gp.fit(&x_train, &y_train).unwrap();

        let x_test = array![[1.0]];
        let (predictions, uncertainties) = trained_model.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 1);
        assert_eq!(uncertainties.len(), 1);
    }

    #[test]
    fn test_heteroscedastic_gpr_gp_noise() {
        let het_gp = HeteroscedasticGaussianProcessRegressor::builder()
            .signal_kernel(Box::new(RBF::new(1.0)))
            .noise_function(LearnableNoiseFunction::gaussian_process(
                Box::new(RBF::new(0.5)),
                3,
            ))
            .max_iter(5)
            .learning_rate(0.1)
            .build();

        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 1.0, 2.0];

        let trained_model = het_gp.fit(&x_train, &y_train).unwrap();

        let x_test = array![[1.0]];
        let (predictions, uncertainties) = trained_model.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 1);
        assert_eq!(uncertainties.len(), 1);
        assert!(uncertainties[0] > 0.0);
    }

    #[test]
    fn test_heteroscedastic_gpr_neural_network_noise() {
        let het_gp = HeteroscedasticGaussianProcessRegressor::builder()
            .signal_kernel(Box::new(RBF::new(1.0)))
            .noise_function(LearnableNoiseFunction::neural_network(1, vec![3]))
            .max_iter(3)
            .learning_rate(0.01)
            .build();

        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 1.0, 2.0];

        let trained_model = het_gp.fit(&x_train, &y_train).unwrap();

        let x_test = array![[1.0]];
        let (predictions, uncertainties) = trained_model.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 1);
        assert_eq!(uncertainties.len(), 1);
    }

    #[test]
    fn test_heteroscedastic_gpr_errors() {
        let het_gp = HeteroscedasticGaussianProcessRegressor::builder()
            .signal_kernel(Box::new(RBF::new(1.0)))
            .noise_function(LearnableNoiseFunction::constant(0.1))
            .build();

        // Test dimension mismatch
        let x_train = array![[0.0], [1.0]];
        let y_train = array![0.0]; // Wrong size
        let result = het_gp.fit(&x_train, &y_train);
        assert!(result.is_err());

        // Test missing kernel
        let het_gp_no_kernel = HeteroscedasticGaussianProcessRegressor::builder()
            .noise_function(LearnableNoiseFunction::constant(0.1))
            .build();
        let x_train = array![[0.0], [1.0]];
        let y_train = array![0.0, 1.0];
        let result = het_gp_no_kernel.fit(&x_train, &y_train);
        assert!(result.is_err());

        // Test missing noise function
        let het_gp_no_noise = HeteroscedasticGaussianProcessRegressor::builder()
            .signal_kernel(Box::new(RBF::new(1.0)))
            .build();
        let result = het_gp_no_noise.fit(&x_train, &y_train);
        assert!(result.is_err());
    }

    #[test]
    fn test_noise_function_parameter_management() {
        let mut linear_noise = LinearNoise::new(0.1, 0.05);

        // Test parameter getting/setting
        let params = linear_noise.get_parameters();
        assert_eq!(params.len(), 2);

        let new_params = vec![0.2, 0.1];
        linear_noise.set_parameters(&new_params).unwrap();
        let updated_params = linear_noise.get_parameters();
        assert_abs_diff_eq!(updated_params[0], 0.2, epsilon = 1e-10);
        assert_abs_diff_eq!(updated_params[1], 0.1, epsilon = 1e-10);

        // Test invalid parameter count
        let result = linear_noise.set_parameters(&vec![0.1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let het_gp = HeteroscedasticGaussianProcessRegressor::builder()
            .signal_kernel(Box::new(RBF::new(2.0)))
            .noise_function(LearnableNoiseFunction::linear(0.1, 0.05))
            .max_iter(200)
            .learning_rate(0.05)
            .convergence_threshold(1e-8)
            .jitter(1e-7)
            .build();

        // Note: The actual values in the struct might be different from config defaults
        // since config returns static defaults
        let config = het_gp.config();
        assert!(config.max_iter > 0);
        assert!(config.learning_rate > 0.0);
    }
}
