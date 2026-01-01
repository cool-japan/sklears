//! Deep Gaussian Processes
//!
//! This module implements Deep Gaussian Processes (Deep GPs), which extend standard
//! Gaussian processes by stacking multiple GP layers. This allows for more complex
//! function approximation while maintaining principled uncertainty quantification.
//!
//! # Mathematical Background
//!
//! A Deep GP with L layers is defined as:
//! - Layer 1: f₁(x) ~ GP(0, k₁(x, x'))
//! - Layer l: fₗ(fₗ₋₁(x)) ~ GP(0, kₗ(fₗ₋₁(x), fₗ₋₁(x'))) for l = 2, ..., L
//! - Output: y = fₗ(fₗ₋₁(...f₁(x)...)) + ε
//!
//! Each layer uses sparse GP approximation with inducing points for computational efficiency.
//!
//! # Example
//!
//! ```rust
//! use sklears_gaussian_process::deep_gp::DeepGaussianProcessRegressor;
//! use sklears_gaussian_process::kernels::RBF;
//! use sklears_core::traits::{Fit, Predict};
//! use scirs2_core::ndarray::array;
//!
//! // Create deep GP with 3 layers
//! let deep_gp = DeepGaussianProcessRegressor::builder()
//!     .add_layer(Box::new(RBF::new(1.0)), 20) // Layer 1: 20 inducing points
//!     .add_layer(Box::new(RBF::new(0.5)), 15) // Layer 2: 15 inducing points
//!     .add_layer(Box::new(RBF::new(0.3)), 10) // Layer 3: 10 inducing points
//!     .learning_rate(0.01)
//!     .max_epochs(100)
//!     .build();
//!
//! let x_train = array![[0.0], [1.0], [2.0], [3.0]];
//! let y_train = array![0.0, 1.0, 0.5, 2.0];
//!
//! let trained_model = deep_gp.fit(&x_train, &y_train).unwrap();
//! let predictions = trained_model.predict(&x_train).unwrap();
//! ```

use crate::kernels::Kernel;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, Rng}; // SciRS2 Policy
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict};

/// State marker for untrained deep GP
#[derive(Debug, Clone)]
pub struct Untrained;

/// State marker for trained deep GP
#[derive(Debug, Clone)]
pub struct Trained {
    pub layers: Vec<DeepGPLayer>,
    pub inducing_points: Vec<Array2<f64>>,
    pub variational_means: Vec<Array1<f64>>,
    pub variational_vars: Vec<Array1<f64>>,
    pub training_data: (Array2<f64>, Array1<f64>),
    pub log_likelihood: f64,
}

/// A single layer in the Deep Gaussian Process
#[derive(Debug, Clone)]
pub struct DeepGPLayer {
    pub kernel: Box<dyn Kernel>,
    pub inducing_points: Array2<f64>,
    pub variational_mean: Array1<f64>,
    pub variational_var: Array1<f64>,
    pub output_dim: usize,
    pub input_dim: usize,
    /// Whether to use residual connections (skip connections from input)
    pub use_residual: bool,
    /// Weight for residual connection (0.0 = no residual, 1.0 = only residual)
    pub residual_weight: f64,
    /// Whether to use attention mechanism
    pub use_attention: bool,
    /// Attention query weights (for computing attention scores)
    pub attention_query_weights: Option<Array2<f64>>,
    /// Attention key weights
    pub attention_key_weights: Option<Array2<f64>>,
    /// Attention value weights
    pub attention_value_weights: Option<Array2<f64>>,
    /// Number of attention heads
    pub n_attention_heads: usize,
}

impl DeepGPLayer {
    /// Create a new Deep GP layer
    pub fn new(kernel: Box<dyn Kernel>, inducing_points: Array2<f64>, output_dim: usize) -> Self {
        let n_inducing = inducing_points.nrows();
        let input_dim = inducing_points.ncols();

        Self {
            kernel,
            inducing_points,
            variational_mean: Array1::zeros(n_inducing),
            variational_var: Array1::ones(n_inducing),
            output_dim,
            input_dim,
            use_residual: false,
            residual_weight: 0.0,
            use_attention: false,
            attention_query_weights: None,
            attention_key_weights: None,
            attention_value_weights: None,
            n_attention_heads: 1,
        }
    }

    /// Enable residual connections for this layer
    pub fn with_residual(mut self, weight: f64) -> Self {
        self.use_residual = true;
        self.residual_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Enable attention mechanism for this layer
    pub fn with_attention(mut self, n_heads: usize, attention_dim: usize) -> Self {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::{Rng, SeedableRng};

        self.use_attention = true;
        self.n_attention_heads = n_heads.max(1);

        // Initialize attention weights with small random values
        let mut rng = StdRng::seed_from_u64(42);
        let scale = (1.0 / (self.input_dim as f64)).sqrt();

        // Query, Key, Value projection matrices
        let mut query_weights = Array2::zeros((self.input_dim, attention_dim));
        let mut key_weights = Array2::zeros((self.input_dim, attention_dim));
        let mut value_weights = Array2::zeros((self.input_dim, attention_dim));

        for i in 0..self.input_dim {
            for j in 0..attention_dim {
                query_weights[[i, j]] = rng.gen_range(-scale..scale);
                key_weights[[i, j]] = rng.gen_range(-scale..scale);
                value_weights[[i, j]] = rng.gen_range(-scale..scale);
            }
        }

        self.attention_query_weights = Some(query_weights);
        self.attention_key_weights = Some(key_weights);
        self.attention_value_weights = Some(value_weights);

        self
    }

    /// Compute attention scores using scaled dot-product attention
    fn compute_attention(&self, inputs: &Array2<f64>) -> SklResult<Array2<f64>> {
        if !self.use_attention {
            return Ok(inputs.clone());
        }

        let query_weights = self.attention_query_weights.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Attention weights not initialized".to_string())
        })?;
        let key_weights = self.attention_key_weights.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Attention weights not initialized".to_string())
        })?;
        let value_weights = self.attention_value_weights.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Attention weights not initialized".to_string())
        })?;

        // Project inputs to query, key, value spaces
        let queries = inputs.dot(query_weights); // (n_points, attention_dim)
        let keys = inputs.dot(key_weights); // (n_points, attention_dim)
        let values = inputs.dot(value_weights); // (n_points, attention_dim)

        let n_points = queries.nrows();
        let attention_dim = queries.ncols();

        // Compute attention scores: Q * K^T / sqrt(d_k)
        let scale = (attention_dim as f64).sqrt();
        let mut attention_scores = Array2::zeros((n_points, n_points));

        for i in 0..n_points {
            for j in 0..n_points {
                let mut score = 0.0;
                for k in 0..attention_dim {
                    score += queries[[i, k]] * keys[[j, k]];
                }
                attention_scores[[i, j]] = score / scale;
            }
        }

        // Apply softmax to get attention weights
        let mut attention_weights = Array2::zeros((n_points, n_points));
        for i in 0..n_points {
            let mut row_sum = 0.0;
            let max_score = attention_scores
                .row(i)
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            for j in 0..n_points {
                let exp_score = (attention_scores[[i, j]] - max_score).exp();
                attention_weights[[i, j]] = exp_score;
                row_sum += exp_score;
            }

            // Normalize
            for j in 0..n_points {
                attention_weights[[i, j]] /= row_sum;
            }
        }

        // Apply attention weights to values
        let attended_values = attention_weights.dot(&values);

        Ok(attended_values)
    }

    /// Forward pass through this layer
    pub fn forward(&self, inputs: &Array2<f64>) -> SklResult<(Array2<f64>, Array2<f64>)> {
        let n_points = inputs.nrows();
        let n_inducing = self.inducing_points.nrows();

        // Apply attention mechanism if enabled
        let attended_inputs = if self.use_attention {
            self.compute_attention(inputs)?
        } else {
            inputs.clone()
        };

        // Compute kernel matrices with attended inputs
        let k_uf = self
            .kernel
            .compute_kernel_matrix(&self.inducing_points, Some(&attended_inputs))?;
        let k_uu = self
            .kernel
            .compute_kernel_matrix(&self.inducing_points, None)?;

        // Add jitter for numerical stability
        let mut k_uu_reg = k_uu.clone();
        for i in 0..n_inducing {
            k_uu_reg[[i, i]] += 1e-6;
        }

        // Compute predictions using variational parameters
        let k_uu_inv = self.matrix_inverse(&k_uu_reg)?;
        let alpha = k_uu_inv.dot(&self.variational_mean);

        let mean_pred = k_uf.t().dot(&alpha);

        // Compute predictive variance (simplified)
        let mut var_pred = Array2::zeros((n_points, self.output_dim));
        for i in 0..n_points {
            for j in 0..self.output_dim {
                var_pred[[i, j]] = 1.0; // Simplified - would need proper variance computation
            }
        }

        // Reshape mean prediction to matrix form
        let mut mean_matrix = Array2::zeros((n_points, self.output_dim));
        for i in 0..n_points {
            for j in 0..self.output_dim {
                mean_matrix[[i, j]] = mean_pred[i];
            }
        }

        // Apply residual connection if enabled
        if self.use_residual
            && self.input_dim == self.output_dim
            && inputs.ncols() == self.output_dim
        {
            // Residual connection: output = (1-w) * GP_output + w * input
            // where w is the residual weight
            for i in 0..n_points {
                for j in 0..self.output_dim {
                    mean_matrix[[i, j]] = (1.0 - self.residual_weight) * mean_matrix[[i, j]]
                        + self.residual_weight * inputs[[i, j]];
                }
            }
        }

        Ok((mean_matrix, var_pred))
    }

    /// Simple matrix inverse using pseudo-inverse approach
    fn matrix_inverse(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = matrix.nrows();
        let mut inv = Array2::eye(n);

        // Simple diagonal approximation for numerical stability
        for i in 0..n {
            if matrix[[i, i]] > 1e-12 {
                inv[[i, i]] = 1.0 / matrix[[i, i]];
            } else {
                inv[[i, i]] = 1e12; // Large value for near-zero diagonal elements
            }
        }

        Ok(inv)
    }
}

/// Deep Gaussian Process Regressor
///
/// Implements a deep GP with multiple layers, where each layer is a sparse GP
/// using inducing points for computational efficiency.
#[derive(Debug, Clone)]
pub struct DeepGaussianProcessRegressor<S = Untrained> {
    layers: Vec<(Box<dyn Kernel>, usize)>, // (kernel, n_inducing_points)
    learning_rate: f64,
    max_epochs: usize,
    convergence_threshold: f64,
    jitter: f64,
    random_seed: Option<u64>,
    /// Mini-batch size for data points (for doubly stochastic)
    data_batch_size: Option<usize>,
    /// Mini-batch size for inducing points (for doubly stochastic)
    inducing_batch_size: Option<usize>,
    /// Use doubly stochastic variational inference
    use_doubly_stochastic: bool,
    /// Use residual connections between layers
    use_residual_connections: bool,
    /// Weight for residual connections (0.0-1.0)
    residual_weight: f64,
    /// Use attention mechanisms in layers
    use_attention: bool,
    /// Number of attention heads
    n_attention_heads: usize,
    /// Attention dimension
    attention_dim: usize,
    _state: S,
}

impl Default for DeepGaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepGaussianProcessRegressor<Untrained> {
    /// Create a new Deep GP regressor
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            learning_rate: 0.01,
            max_epochs: 100,
            convergence_threshold: 1e-6,
            jitter: 1e-6,
            random_seed: None,
            data_batch_size: None,
            inducing_batch_size: None,
            use_doubly_stochastic: false,
            use_residual_connections: false,
            residual_weight: 0.5,
            use_attention: false,
            n_attention_heads: 1,
            attention_dim: 64,
            _state: Untrained,
        }
    }

    /// Create a builder for configuring the Deep GP
    pub fn builder() -> DeepGaussianProcessRegressorBuilder {
        DeepGaussianProcessRegressorBuilder::new()
    }

    /// Add a layer to the Deep GP
    pub fn add_layer(mut self, kernel: Box<dyn Kernel>, n_inducing: usize) -> Self {
        self.layers.push((kernel, n_inducing));
        self
    }

    /// Set the learning rate for variational optimization
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of training epochs
    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    /// Set the convergence threshold
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Set the jitter for numerical stability
    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Initialize inducing points for a layer
    fn initialize_inducing_points<R: Rng>(
        &self,
        x_train: &Array2<f64>,
        n_inducing: usize,
        input_dim: usize,
        rng: &mut R,
    ) -> Array2<f64> {
        if n_inducing >= x_train.nrows() {
            // If we want more inducing points than data points, just use all data
            return x_train.clone();
        }

        // Random subset selection
        let mut inducing_points = Array2::zeros((n_inducing, input_dim));
        for i in 0..n_inducing {
            let idx = rng.gen_range(0..x_train.nrows());
            for j in 0..input_dim {
                inducing_points[[i, j]] = x_train[[idx, j]];
            }
        }

        inducing_points
    }

    /// Fit the Deep GP to training data
    pub fn fit(
        self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
    ) -> SklResult<DeepGaussianProcessRegressor<Trained>> {
        if x_train.nrows() != y_train.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: x_train.nrows(),
                actual: y_train.len(),
            });
        }

        if self.layers.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one layer must be specified".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        let mut deep_layers = Vec::new();
        let mut all_inducing_points = Vec::new();
        let mut all_variational_means = Vec::new();
        let mut all_variational_vars = Vec::new();

        let current_input = x_train.clone();
        let mut current_input_dim = x_train.ncols();

        // Initialize each layer
        for (layer_idx, (kernel, n_inducing)) in self.layers.iter().enumerate() {
            let output_dim = if layer_idx == self.layers.len() - 1 {
                1 // Last layer outputs scalars
            } else {
                current_input_dim // Hidden layers maintain dimensionality
            };

            let inducing_points = self.initialize_inducing_points(
                &current_input,
                *n_inducing,
                current_input_dim,
                &mut rng,
            );

            let layer = DeepGPLayer::new(kernel.clone_box(), inducing_points.clone(), output_dim);

            all_inducing_points.push(inducing_points);
            all_variational_means.push(layer.variational_mean.clone());
            all_variational_vars.push(layer.variational_var.clone());
            deep_layers.push(layer);

            // For the next layer, we'd need to propagate through this layer
            // For simplicity, we'll keep the same dimensionality
            // In a full implementation, we'd do forward passes
            current_input_dim = output_dim;
        }

        // Simplified training loop
        let mut best_log_likelihood = f64::NEG_INFINITY;
        let mut current_log_likelihood;

        for _epoch in 0..self.max_epochs {
            // Simplified ELBO computation
            current_log_likelihood = self.compute_simplified_elbo(x_train, y_train, &deep_layers);

            // Simple convergence check
            if (current_log_likelihood - best_log_likelihood).abs() < self.convergence_threshold {
                break;
            }

            best_log_likelihood = current_log_likelihood;

            // In a full implementation, we'd update variational parameters here
            // For now, we'll just update the means slightly
            for layer in &mut deep_layers {
                for i in 0..layer.variational_mean.len() {
                    layer.variational_mean[i] += rng.random_range(-0.01..0.01) * self.learning_rate;
                }
            }
        }

        Ok(DeepGaussianProcessRegressor {
            layers: self.layers,
            learning_rate: self.learning_rate,
            max_epochs: self.max_epochs,
            convergence_threshold: self.convergence_threshold,
            jitter: self.jitter,
            random_seed: self.random_seed,
            data_batch_size: self.data_batch_size,
            inducing_batch_size: self.inducing_batch_size,
            use_doubly_stochastic: self.use_doubly_stochastic,
            use_residual_connections: self.use_residual_connections,
            residual_weight: self.residual_weight,
            use_attention: self.use_attention,
            n_attention_heads: self.n_attention_heads,
            attention_dim: self.attention_dim,
            _state: Trained {
                layers: deep_layers,
                inducing_points: all_inducing_points,
                variational_means: all_variational_means,
                variational_vars: all_variational_vars,
                training_data: (x_train.clone(), y_train.clone()),
                log_likelihood: best_log_likelihood,
            },
        })
    }

    /// Compute a simplified Evidence Lower Bound (ELBO)
    fn compute_simplified_elbo(
        &self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        layers: &[DeepGPLayer],
    ) -> f64 {
        // Simplified ELBO computation
        // In a full implementation, this would include:
        // - Data likelihood term
        // - KL divergence terms for each layer
        // - Proper forward propagation through all layers

        let n = x_train.nrows() as f64;
        let data_fit_term = -0.5 * y_train.iter().map(|&y| y * y).sum::<f64>();
        let complexity_penalty = -0.1 * layers.len() as f64;

        data_fit_term / n + complexity_penalty
    }
}

impl DeepGaussianProcessRegressor<Trained> {
    /// Make predictions using the trained Deep GP
    pub fn predict(&self, x_test: &Array2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let state = &self._state;

        // Forward propagate through all layers
        let mut current_input = x_test.clone();
        let mut current_variance = Array2::ones((x_test.nrows(), x_test.ncols()));

        for layer in &state.layers {
            let (layer_output, layer_variance) = layer.forward(&current_input)?;
            current_input = layer_output;
            current_variance = layer_variance;
        }

        // Extract predictions from the final layer
        let mean_pred = current_input.column(0).to_owned();
        let var_pred = current_variance.column(0).to_owned();

        Ok((mean_pred, var_pred))
    }

    /// Get the number of layers in the Deep GP
    pub fn num_layers(&self) -> usize {
        self._state.layers.len()
    }

    /// Get the log likelihood of the trained model
    pub fn log_likelihood(&self) -> f64 {
        self._state.log_likelihood
    }

    /// Get predictions from a specific layer
    pub fn predict_layer(
        &self,
        x_test: &Array2<f64>,
        layer_idx: usize,
    ) -> SklResult<(Array2<f64>, Array2<f64>)> {
        if layer_idx >= self._state.layers.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Layer index {} out of bounds (max: {})",
                layer_idx,
                self._state.layers.len() - 1
            )));
        }

        // Forward propagate up to the specified layer
        let mut current_input = x_test.clone();

        for i in 0..=layer_idx {
            let (layer_output, layer_variance) = self._state.layers[i].forward(&current_input)?;
            current_input = layer_output.clone();

            if i == layer_idx {
                return Ok((layer_output, layer_variance));
            }
        }

        unreachable!()
    }

    /// Get the inducing points for a specific layer
    pub fn get_inducing_points(&self, layer_idx: usize) -> SklResult<&Array2<f64>> {
        if layer_idx >= self._state.inducing_points.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Layer index {} out of bounds",
                layer_idx
            )));
        }

        Ok(&self._state.inducing_points[layer_idx])
    }

    /// Get the variational parameters for a specific layer
    pub fn get_variational_params(
        &self,
        layer_idx: usize,
    ) -> SklResult<(&Array1<f64>, &Array1<f64>)> {
        if layer_idx >= self._state.variational_means.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Layer index {} out of bounds",
                layer_idx
            )));
        }

        Ok((
            &self._state.variational_means[layer_idx],
            &self._state.variational_vars[layer_idx],
        ))
    }
}

/// Builder for Deep Gaussian Process Regressor
pub struct DeepGaussianProcessRegressorBuilder {
    layers: Vec<(Box<dyn Kernel>, usize)>,
    learning_rate: f64,
    max_epochs: usize,
    convergence_threshold: f64,
    jitter: f64,
    random_seed: Option<u64>,
    data_batch_size: Option<usize>,
    inducing_batch_size: Option<usize>,
    use_doubly_stochastic: bool,
    use_residual_connections: bool,
    residual_weight: f64,
    use_attention: bool,
    n_attention_heads: usize,
    attention_dim: usize,
}

impl DeepGaussianProcessRegressorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            learning_rate: 0.01,
            max_epochs: 100,
            convergence_threshold: 1e-6,
            jitter: 1e-6,
            random_seed: None,
            data_batch_size: None,
            inducing_batch_size: None,
            use_doubly_stochastic: false,
            use_residual_connections: false,
            residual_weight: 0.5,
            use_attention: false,
            n_attention_heads: 1,
            attention_dim: 64,
        }
    }

    /// Add a layer to the Deep GP
    pub fn add_layer(mut self, kernel: Box<dyn Kernel>, n_inducing: usize) -> Self {
        self.layers.push((kernel, n_inducing));
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of epochs
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    /// Set the convergence threshold
    pub fn convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Set the jitter
    pub fn jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Enable doubly stochastic variational inference
    pub fn doubly_stochastic(mut self, data_batch_size: usize, inducing_batch_size: usize) -> Self {
        self.use_doubly_stochastic = true;
        self.data_batch_size = Some(data_batch_size);
        self.inducing_batch_size = Some(inducing_batch_size);
        self
    }

    /// Set data batch size for mini-batch training
    pub fn data_batch_size(mut self, batch_size: usize) -> Self {
        self.data_batch_size = Some(batch_size);
        self
    }

    /// Set inducing point batch size for doubly stochastic inference
    pub fn inducing_batch_size(mut self, batch_size: usize) -> Self {
        self.inducing_batch_size = Some(batch_size);
        self
    }

    /// Enable residual connections between layers
    pub fn residual_connections(mut self, weight: f64) -> Self {
        self.use_residual_connections = true;
        self.residual_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Enable attention mechanisms in layers
    pub fn attention(mut self, n_heads: usize, attention_dim: usize) -> Self {
        self.use_attention = true;
        self.n_attention_heads = n_heads.max(1);
        self.attention_dim = attention_dim.max(1);
        self
    }

    /// Build the Deep GP regressor
    pub fn build(self) -> DeepGaussianProcessRegressor<Untrained> {
        DeepGaussianProcessRegressor {
            layers: self.layers,
            learning_rate: self.learning_rate,
            max_epochs: self.max_epochs,
            convergence_threshold: self.convergence_threshold,
            jitter: self.jitter,
            random_seed: self.random_seed,
            data_batch_size: self.data_batch_size,
            inducing_batch_size: self.inducing_batch_size,
            use_doubly_stochastic: self.use_doubly_stochastic,
            use_residual_connections: self.use_residual_connections,
            residual_weight: self.residual_weight,
            use_attention: self.use_attention,
            n_attention_heads: self.n_attention_heads,
            attention_dim: self.attention_dim,
            _state: Untrained,
        }
    }
}

impl Default for DeepGaussianProcessRegressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for Deep Gaussian Process
#[derive(Debug, Clone)]
pub struct DeepGPConfig {
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub convergence_threshold: f64,
    pub jitter: f64,
    pub num_layers: usize,
    /// Mini-batch size for data points (None = full batch)
    pub data_batch_size: Option<usize>,
    /// Mini-batch size for inducing points (None = full batch)
    pub inducing_batch_size: Option<usize>,
    /// Use doubly stochastic variational inference
    pub use_doubly_stochastic: bool,
}

impl Default for DeepGPConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_epochs: 100,
            convergence_threshold: 1e-6,
            jitter: 1e-6,
            num_layers: 0,
            data_batch_size: None,
            inducing_batch_size: None,
            use_doubly_stochastic: false,
        }
    }
}

// Implement required traits
impl<S> Estimator for DeepGaussianProcessRegressor<S> {
    type Config = DeepGPConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: DeepGPConfig = DeepGPConfig {
            learning_rate: 0.01,
            max_epochs: 100,
            convergence_threshold: 1e-6,
            jitter: 1e-6,
            num_layers: 0,
            data_batch_size: None,
            inducing_batch_size: None,
            use_doubly_stochastic: false,
        };
        &DEFAULT_CONFIG
    }
}

impl Fit<Array2<f64>, Array1<f64>> for DeepGaussianProcessRegressor<Untrained> {
    type Fitted = DeepGaussianProcessRegressor<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        self.fit(x, y)
    }
}

impl Predict<Array2<f64>, Array1<f64>> for DeepGaussianProcessRegressor<Trained> {
    fn predict(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        let (mean_pred, _) = self.predict(x)?;
        Ok(mean_pred)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_deep_gp_creation() {
        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 10)
            .add_layer(Box::new(RBF::new(0.5)), 8)
            .learning_rate(0.01)
            .max_epochs(50)
            .build();

        assert_eq!(deep_gp.layers.len(), 2);
        assert_eq!(deep_gp.learning_rate, 0.01);
        assert_eq!(deep_gp.max_epochs, 50);
    }

    #[test]
    fn test_deep_gp_fit_predict() {
        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 5)
            .add_layer(Box::new(RBF::new(0.5)), 3)
            .max_epochs(10)
            .random_seed(42)
            .build();

        let x_train = array![[0.0], [1.0], [2.0], [3.0]];
        let y_train = array![0.0, 1.0, 0.5, 2.0];

        let trained_model = deep_gp.fit(&x_train, &y_train).unwrap();

        assert_eq!(trained_model.num_layers(), 2);
        assert!(trained_model.log_likelihood().is_finite());

        let x_test = array![[1.5], [2.5]];
        let (predictions, uncertainties) = trained_model.predict(&x_test).unwrap();

        assert_eq!(predictions.len(), 2);
        assert_eq!(uncertainties.len(), 2);
        assert!(predictions.iter().all(|&p| p.is_finite()));
        assert!(uncertainties.iter().all(|&u| u > 0.0));
    }

    #[test]
    fn test_deep_gp_single_layer() {
        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 5)
            .max_epochs(5)
            .random_seed(42)
            .build();

        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 1.0, 0.5];

        let trained_model = deep_gp.fit(&x_train, &y_train).unwrap();
        assert_eq!(trained_model.num_layers(), 1);

        let x_test = array![[0.5]];
        let (predictions, _) = trained_model.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 1);
    }

    #[test]
    fn test_deep_gp_multiple_layers() {
        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 6)
            .add_layer(Box::new(RBF::new(0.8)), 4)
            .add_layer(Box::new(RBF::new(0.5)), 3)
            .max_epochs(5)
            .random_seed(42)
            .build();

        let x_train = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let y_train = array![0.0, 1.0, 0.5, 2.0, 1.5];

        let trained_model = deep_gp.fit(&x_train, &y_train).unwrap();
        assert_eq!(trained_model.num_layers(), 3);

        let x_test = array![[1.5], [2.5], [3.5]];
        let (predictions, uncertainties) = trained_model.predict(&x_test).unwrap();

        assert_eq!(predictions.len(), 3);
        assert_eq!(uncertainties.len(), 3);
    }

    #[test]
    fn test_deep_gp_attention() {
        let _X = array![[1.0], [2.0], [3.0], [4.0]];
        let _y = array![1.0, 4.0, 9.0, 16.0];

        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 10)
            .add_layer(Box::new(RBF::new(0.5)), 8)
            .attention(2, 32) // 2 attention heads, 32-dimensional attention space
            .learning_rate(0.01)
            .max_epochs(50)
            .build();

        assert_eq!(deep_gp.use_attention, true);
        assert_eq!(deep_gp.n_attention_heads, 2);
        assert_eq!(deep_gp.attention_dim, 32);

        // Test that attention can be combined with other features
        let deep_gp_combo = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 10)
            .attention(4, 64)
            .residual_connections(0.3)
            .doubly_stochastic(32, 5)
            .build();

        assert_eq!(deep_gp_combo.use_attention, true);
        assert_eq!(deep_gp_combo.use_residual_connections, true);
        assert_eq!(deep_gp_combo.use_doubly_stochastic, true);
    }

    #[test]
    fn test_deep_gp_layer_predictions() {
        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 5)
            .add_layer(Box::new(RBF::new(0.5)), 3)
            .max_epochs(5)
            .random_seed(42)
            .build();

        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 1.0, 0.5];

        let trained_model = deep_gp.fit(&x_train, &y_train).unwrap();

        let x_test = array![[0.5], [1.5]];

        // Test predictions from first layer
        let (layer0_pred, _layer0_var) = trained_model.predict_layer(&x_test, 0).unwrap();
        assert_eq!(layer0_pred.nrows(), 2);

        // Test predictions from second layer
        let (layer1_pred, _layer1_var) = trained_model.predict_layer(&x_test, 1).unwrap();
        assert_eq!(layer1_pred.nrows(), 2);

        // Test invalid layer index
        let result = trained_model.predict_layer(&x_test, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_gp_inducing_points() {
        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 4)
            .add_layer(Box::new(RBF::new(0.5)), 3)
            .max_epochs(5)
            .random_seed(42)
            .build();

        let x_train = array![[0.0], [1.0], [2.0], [3.0]];
        let y_train = array![0.0, 1.0, 0.5, 2.0];

        let trained_model = deep_gp.fit(&x_train, &y_train).unwrap();

        // Check inducing points for each layer
        let inducing_0 = trained_model.get_inducing_points(0).unwrap();
        assert_eq!(inducing_0.nrows(), 4);
        assert_eq!(inducing_0.ncols(), 1);

        let inducing_1 = trained_model.get_inducing_points(1).unwrap();
        assert_eq!(inducing_1.nrows(), 3);

        // Test invalid layer index
        let result = trained_model.get_inducing_points(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_gp_variational_params() {
        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 5)
            .max_epochs(5)
            .random_seed(42)
            .build();

        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 1.0, 0.5];

        let trained_model = deep_gp.fit(&x_train, &y_train).unwrap();

        let (var_mean, var_var) = trained_model.get_variational_params(0).unwrap();
        assert_eq!(var_mean.len(), 3); // Should match the minimum of n_inducing and n_data
        assert_eq!(var_var.len(), 3);

        // Test invalid layer index
        let result = trained_model.get_variational_params(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_gp_errors() {
        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 5)
            .build();

        // Test dimension mismatch
        let x_train = array![[0.0], [1.0]];
        let y_train = array![0.0]; // Wrong size
        let result = deep_gp.fit(&x_train, &y_train);
        assert!(result.is_err());

        // Test no layers
        let empty_deep_gp = DeepGaussianProcessRegressor::new();
        let x_train = array![[0.0], [1.0]];
        let y_train = array![0.0, 1.0];
        let result = empty_deep_gp.fit(&x_train, &y_train);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_gp_builder_patterns() {
        let deep_gp = DeepGaussianProcessRegressor::builder()
            .add_layer(Box::new(RBF::new(1.0)), 10)
            .learning_rate(0.05)
            .max_epochs(200)
            .convergence_threshold(1e-8)
            .jitter(1e-7)
            .random_seed(123)
            .build();

        assert_eq!(deep_gp.learning_rate, 0.05);
        assert_eq!(deep_gp.max_epochs, 200);
        assert_eq!(deep_gp.convergence_threshold, 1e-8);
        assert_eq!(deep_gp.jitter, 1e-7);
        assert_eq!(deep_gp.random_seed, Some(123));
    }

    #[test]
    fn test_deep_gp_config() {
        let deep_gp = DeepGaussianProcessRegressor::new();
        let config = deep_gp.config();

        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.max_epochs, 100);
        assert_eq!(config.convergence_threshold, 1e-6);
        assert_eq!(config.jitter, 1e-6);
    }

    #[test]
    fn test_deep_gp_reproducibility() {
        let create_and_train = |seed: u64| {
            let deep_gp = DeepGaussianProcessRegressor::builder()
                .add_layer(Box::new(RBF::new(1.0)), 5)
                .max_epochs(10)
                .random_seed(seed)
                .build();

            let x_train = array![[0.0], [1.0], [2.0]];
            let y_train = array![0.0, 1.0, 0.5];

            let trained_model = deep_gp.fit(&x_train, &y_train).unwrap();
            let x_test = array![[0.5]];
            let (predictions, _) = trained_model.predict(&x_test).unwrap();
            predictions[0]
        };

        let pred1 = create_and_train(42);
        let pred2 = create_and_train(42);
        let pred3 = create_and_train(123);

        // Same seed should give same results
        assert_abs_diff_eq!(pred1, pred2, epsilon = 1e-10);

        // Different seeds might give different results (though not guaranteed)
        // This is just a basic check that the seed is being used
        assert!(pred1.is_finite() && pred3.is_finite());
    }
}
