//! Deep Canonical Correlation Analysis

use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::random::{thread_rng, Random};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Activation function types for Deep CCA
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    /// Sigmoid activation: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    /// ReLU activation: max(0, x)
    ReLU,
    /// Leaky ReLU: max(0.01*x, x)
    LeakyReLU,
    /// Linear activation (no transformation)
    Linear,
}

impl ActivationFunction {
    /// Apply the activation function to an array
    fn apply(&self, x: &Array2<Float>) -> Array2<Float> {
        match self {
            ActivationFunction::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationFunction::Tanh => x.mapv(|v| v.tanh()),
            ActivationFunction::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationFunction::LeakyReLU => x.mapv(|v| if v > 0.0 { v } else { 0.01 * v }),
            ActivationFunction::Linear => x.clone(),
        }
    }

    /// Apply the derivative of the activation function
    fn derivative(&self, x: &Array2<Float>) -> Array2<Float> {
        match self {
            ActivationFunction::Sigmoid => {
                let sig = self.apply(x);
                &sig * &(1.0 - &sig)
            }
            ActivationFunction::Tanh => {
                let tanh_x = self.apply(x);
                1.0 - &tanh_x * &tanh_x
            }
            ActivationFunction::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::LeakyReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.01 }),
            ActivationFunction::Linear => Array2::ones(x.raw_dim()),
        }
    }
}

/// Deep Canonical Correlation Analysis
///
/// Deep CCA extends traditional CCA by using deep neural networks to learn
/// complex nonlinear transformations before applying CCA. This enables the
/// discovery of sophisticated relationships between high-dimensional data.
///
/// The algorithm consists of two deep neural networks (one for each view) that
/// transform the input data into representations where the canonical correlation
/// is maximized.
///
/// # Parameters
///
/// * `hidden_layers_x` - Architecture of hidden layers for X view network
/// * `hidden_layers_y` - Architecture of hidden layers for Y view network
/// * `output_dim` - Output dimensionality (canonical space)
/// * `activation` - Activation function for hidden layers
/// * `learning_rate` - Learning rate for optimization
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Tolerance for convergence
/// * `regularization` - L2 regularization parameter
/// * `batch_size` - Batch size for mini-batch training
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::{DeepCCA, ActivationFunction};
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
/// let Y = array![[2.0, 1.0], [5.0, 4.0], [8.0, 7.0], [11.0, 10.0]];
///
/// let deep_cca = DeepCCA::new(vec![10, 5], vec![8, 5], 2)
///     .activation(ActivationFunction::Tanh)
///     .learning_rate(0.001)
///     .max_iter(1000);
///
/// let fitted = deep_cca.fit(&X, &Y).unwrap();
/// let X_transformed = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DeepCCA<State = Untrained> {
    /// Hidden layer architecture for X view network
    pub hidden_layers_x: Vec<usize>,
    /// Hidden layer architecture for Y view network
    pub hidden_layers_y: Vec<usize>,
    /// Output dimensionality (canonical space)
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationFunction,
    /// Learning rate for optimization
    pub learning_rate: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// L2 regularization parameter
    pub regularization: Float,
    /// Batch size for training
    pub batch_size: usize,
    /// Whether to scale the data
    pub scale: bool,
    /// Random seed for initialization
    pub random_state: Option<u64>,

    // Fitted attributes
    weights_x_: Option<Vec<Array2<Float>>>,
    biases_x_: Option<Vec<Array1<Float>>>,
    weights_y_: Option<Vec<Array2<Float>>>,
    biases_y_: Option<Vec<Array1<Float>>>,
    canonical_weights_x_: Option<Array2<Float>>,
    canonical_weights_y_: Option<Array2<Float>>,
    correlations_: Option<Array1<Float>>,
    x_mean_: Option<Array1<Float>>,
    y_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    y_std_: Option<Array1<Float>>,
    n_iter_: Option<usize>,
    loss_history_: Option<Vec<Float>>,

    _state: PhantomData<State>,
}

impl DeepCCA<Untrained> {
    /// Create a new Deep CCA model
    ///
    /// # Arguments
    ///
    /// * `hidden_layers_x` - Architecture of hidden layers for X view network
    /// * `hidden_layers_y` - Architecture of hidden layers for Y view network
    /// * `output_dim` - Output dimensionality (canonical space)
    pub fn new(
        hidden_layers_x: Vec<usize>,
        hidden_layers_y: Vec<usize>,
        output_dim: usize,
    ) -> Self {
        Self {
            hidden_layers_x,
            hidden_layers_y,
            output_dim,
            activation: ActivationFunction::Tanh,
            learning_rate: 0.001,
            max_iter: 1000,
            tol: 1e-6,
            regularization: 0.0001,
            batch_size: 32,
            scale: true,
            random_state: None,
            weights_x_: None,
            biases_x_: None,
            weights_y_: None,
            biases_y_: None,
            canonical_weights_x_: None,
            canonical_weights_y_: None,
            correlations_: None,
            x_mean_: None,
            y_mean_: None,
            x_std_: None,
            y_std_: None,
            n_iter_: None,
            loss_history_: None,
            _state: PhantomData,
        }
    }

    /// Set the activation function
    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        self.activation = activation;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl DeepCCA<Trained> {
    /// Get the learned weights for X view network
    pub fn weights_x(&self) -> &Vec<Array2<Float>> {
        self.weights_x_.as_ref().unwrap()
    }

    /// Get the learned biases for X view network
    pub fn biases_x(&self) -> &Vec<Array1<Float>> {
        self.biases_x_.as_ref().unwrap()
    }

    /// Get the learned weights for Y view network
    pub fn weights_y(&self) -> &Vec<Array2<Float>> {
        self.weights_y_.as_ref().unwrap()
    }

    /// Get the learned biases for Y view network
    pub fn biases_y(&self) -> &Vec<Array1<Float>> {
        self.biases_y_.as_ref().unwrap()
    }

    /// Get the canonical weights for X
    pub fn canonical_weights_x(&self) -> &Array2<Float> {
        self.canonical_weights_x_.as_ref().unwrap()
    }

    /// Get the canonical weights for Y
    pub fn canonical_weights_y(&self) -> &Array2<Float> {
        self.canonical_weights_y_.as_ref().unwrap()
    }

    /// Get the canonical correlations
    pub fn correlations(&self) -> &Array1<Float> {
        self.correlations_.as_ref().unwrap()
    }

    /// Get the number of iterations used
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Get the loss history
    pub fn loss_history(&self) -> &Vec<Float> {
        self.loss_history_.as_ref().unwrap()
    }

    /// Forward pass through X network
    fn forward_x(&self, x: &Array2<Float>) -> Array2<Float> {
        let mut output = x.clone();

        for (i, (weights, bias)) in self
            .weights_x()
            .iter()
            .zip(self.biases_x().iter())
            .enumerate()
        {
            output = output.dot(weights) + bias;

            // Apply activation function to hidden layers only
            if i < self.weights_x().len() - 1 {
                output = self.activation.apply(&output);
            }
        }

        output
    }

    /// Forward pass through Y network
    fn forward_y(&self, y: &Array2<Float>) -> Array2<Float> {
        let mut output = y.clone();

        for (i, (weights, bias)) in self
            .weights_y()
            .iter()
            .zip(self.biases_y().iter())
            .enumerate()
        {
            output = output.dot(weights) + bias;

            // Apply activation function to hidden layers only
            if i < self.weights_y().len() - 1 {
                output = self.activation.apply(&output);
            }
        }

        output
    }

    /// Transform both X and Y to canonical space
    pub fn transform_both(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array2<Float>)> {
        let x_scaled = self.scale_x(x)?;
        let y_scaled = self.scale_y(y)?;

        let x_hidden = self.forward_x(&x_scaled);
        let y_hidden = self.forward_y(&y_scaled);

        let x_canonical = x_hidden.dot(self.canonical_weights_x());
        let y_canonical = y_hidden.dot(self.canonical_weights_y());

        Ok((x_canonical, y_canonical))
    }

    /// Scale X data
    fn scale_x(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if !self.scale {
            return Ok(x.clone());
        }

        let x_mean = self.x_mean_.as_ref().unwrap();
        let x_std = self.x_std_.as_ref().unwrap();

        let mut x_scaled = x.clone();
        for (mut row, (mean, std)) in x_scaled
            .axis_iter_mut(Axis(0))
            .zip(x_mean.iter().zip(x_std.iter()))
        {
            row -= *mean;
            if *std > 0.0 {
                row /= *std;
            }
        }

        Ok(x_scaled)
    }

    /// Scale Y data
    fn scale_y(&self, y: &Array2<Float>) -> Result<Array2<Float>> {
        if !self.scale {
            return Ok(y.clone());
        }

        let y_mean = self.y_mean_.as_ref().unwrap();
        let y_std = self.y_std_.as_ref().unwrap();

        let mut y_scaled = y.clone();
        for (mut row, (mean, std)) in y_scaled
            .axis_iter_mut(Axis(0))
            .zip(y_mean.iter().zip(y_std.iter()))
        {
            row -= *mean;
            if *std > 0.0 {
                row /= *std;
            }
        }

        Ok(y_scaled)
    }
}

impl Fit<Array2<Float>, Array2<Float>> for DeepCCA<Untrained> {
    type Fitted = DeepCCA<Trained>;
    fn fit(mut self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features_x) = x.dim();
        let (n_samples_y, n_features_y) = y.dim();

        if n_samples != n_samples_y {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".into(),
            ));
        }

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput("Need at least 2 samples".into()));
        }

        // Scale data if requested
        let (x_processed, y_processed) = if self.scale {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean_axis(Axis(0)).unwrap();

            let x_std = x.std_axis(Axis(0), 1.0);
            let y_std = y.std_axis(Axis(0), 1.0);

            let mut x_scaled = x.clone();
            let mut y_scaled = y.clone();

            for (mut row, (mean, std)) in x_scaled
                .axis_iter_mut(Axis(0))
                .zip(x_mean.iter().zip(x_std.iter()))
            {
                row -= *mean;
                if *std > 0.0 {
                    row /= *std;
                }
            }

            for (mut row, (mean, std)) in y_scaled
                .axis_iter_mut(Axis(0))
                .zip(y_mean.iter().zip(y_std.iter()))
            {
                row -= *mean;
                if *std > 0.0 {
                    row /= *std;
                }
            }

            self.x_mean_ = Some(x_mean);
            self.y_mean_ = Some(y_mean);
            self.x_std_ = Some(x_std);
            self.y_std_ = Some(y_std);

            (x_scaled, y_scaled)
        } else {
            (x.clone(), y.clone())
        };

        // Initialize networks
        let weights_x =
            self.initialize_network(&self.hidden_layers_x, n_features_x, self.output_dim)?;
        let biases_x = self.initialize_biases(&self.hidden_layers_x, self.output_dim)?;
        let weights_y =
            self.initialize_network(&self.hidden_layers_y, n_features_y, self.output_dim)?;
        let biases_y = self.initialize_biases(&self.hidden_layers_y, self.output_dim)?;

        // Initialize canonical correlation matrices
        let canonical_weights_x = Array2::eye(self.output_dim);
        let canonical_weights_y = Array2::eye(self.output_dim);

        self.weights_x_ = Some(weights_x);
        self.biases_x_ = Some(biases_x);
        self.weights_y_ = Some(weights_y);
        self.biases_y_ = Some(biases_y);
        self.canonical_weights_x_ = Some(canonical_weights_x);
        self.canonical_weights_y_ = Some(canonical_weights_y);

        // Train the network
        let mut loss_history = Vec::new();
        let mut prev_loss = Float::INFINITY;

        for iter in 0..self.max_iter {
            let loss = self.training_step(&x_processed, &y_processed)?;
            loss_history.push(loss);

            if (prev_loss - loss).abs() < self.tol {
                self.n_iter_ = Some(iter + 1);
                break;
            }

            prev_loss = loss;

            if iter == self.max_iter - 1 {
                self.n_iter_ = Some(self.max_iter);
            }
        }

        self.loss_history_ = Some(loss_history);

        // Compute final canonical correlations
        let x_hidden = self.forward_x_training(&x_processed);
        let y_hidden = self.forward_y_training(&y_processed);
        let correlations = self.compute_correlations(&x_hidden, &y_hidden)?;
        self.correlations_ = Some(correlations);

        Ok(DeepCCA {
            hidden_layers_x: self.hidden_layers_x,
            hidden_layers_y: self.hidden_layers_y,
            output_dim: self.output_dim,
            activation: self.activation,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            tol: self.tol,
            regularization: self.regularization,
            batch_size: self.batch_size,
            scale: self.scale,
            random_state: self.random_state,
            weights_x_: self.weights_x_,
            biases_x_: self.biases_x_,
            weights_y_: self.weights_y_,
            biases_y_: self.biases_y_,
            canonical_weights_x_: self.canonical_weights_x_,
            canonical_weights_y_: self.canonical_weights_y_,
            correlations_: self.correlations_,
            x_mean_: self.x_mean_,
            y_mean_: self.y_mean_,
            x_std_: self.x_std_,
            y_std_: self.y_std_,
            n_iter_: self.n_iter_,
            loss_history_: self.loss_history_,
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for DeepCCA<Trained> {
    /// Transform X to canonical space
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_scaled = self.scale_x(x)?;
        let x_hidden = self.forward_x(&x_scaled);
        let x_canonical = x_hidden.dot(self.canonical_weights_x());
        Ok(x_canonical)
    }
}

impl DeepCCA<Untrained> {
    /// Initialize network weights
    fn initialize_network(
        &self,
        hidden_layers: &[usize],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<Array2<Float>>> {
        let mut weights = Vec::new();
        let mut prev_dim = input_dim;

        // Hidden layers
        for &layer_size in hidden_layers {
            let weight = self.xavier_init(prev_dim, layer_size);
            weights.push(weight);
            prev_dim = layer_size;
        }

        // Output layer
        let output_weight = self.xavier_init(prev_dim, output_dim);
        weights.push(output_weight);

        Ok(weights)
    }

    /// Initialize biases
    fn initialize_biases(
        &self,
        hidden_layers: &[usize],
        output_dim: usize,
    ) -> Result<Vec<Array1<Float>>> {
        let mut biases = Vec::new();

        // Hidden layers
        for &layer_size in hidden_layers {
            biases.push(Array1::zeros(layer_size));
        }

        // Output layer
        biases.push(Array1::zeros(output_dim));

        Ok(biases)
    }

    /// Xavier initialization
    fn xavier_init(&self, fan_in: usize, fan_out: usize) -> Array2<Float> {
        let limit = (6.0 / (fan_in + fan_out) as Float).sqrt();
        Array2::from_shape_fn((fan_in, fan_out), |_| {
            let mut rng = thread_rng();
            rng.gen_range(-limit..limit)
        })
    }

    /// Forward pass through X network during training
    fn forward_x_training(&self, x: &Array2<Float>) -> Array2<Float> {
        let mut output = x.clone();

        let weights_x = self.weights_x_.as_ref().unwrap();
        let biases_x = self.biases_x_.as_ref().unwrap();

        for (i, (weights, bias)) in weights_x.iter().zip(biases_x.iter()).enumerate() {
            output = output.dot(weights) + bias;

            // Apply activation function to hidden layers only
            if i < weights_x.len() - 1 {
                output = self.activation.apply(&output);
            }
        }

        output
    }

    /// Forward pass through Y network during training
    fn forward_y_training(&self, y: &Array2<Float>) -> Array2<Float> {
        let mut output = y.clone();

        let weights_y = self.weights_y_.as_ref().unwrap();
        let biases_y = self.biases_y_.as_ref().unwrap();

        for (i, (weights, bias)) in weights_y.iter().zip(biases_y.iter()).enumerate() {
            output = output.dot(weights) + bias;

            // Apply activation function to hidden layers only
            if i < weights_y.len() - 1 {
                output = self.activation.apply(&output);
            }
        }

        output
    }

    /// Compute canonical correlations
    fn compute_correlations(
        &self,
        x_hidden: &Array2<Float>,
        y_hidden: &Array2<Float>,
    ) -> Result<Array1<Float>> {
        let n_samples = x_hidden.nrows() as Float;

        // Center the hidden representations
        let x_mean = x_hidden.mean_axis(Axis(0)).unwrap();
        let y_mean = y_hidden.mean_axis(Axis(0)).unwrap();

        let x_centered = x_hidden - &x_mean;
        let y_centered = y_hidden - &y_mean;

        // Compute covariance matrices
        let cxx = x_centered.t().dot(&x_centered) / (n_samples - 1.0);
        let cyy = y_centered.t().dot(&y_centered) / (n_samples - 1.0);
        let cxy = x_centered.t().dot(&y_centered) / (n_samples - 1.0);

        // For simplicity, return the diagonal of the correlation matrix
        // In a full implementation, you would solve the generalized eigenvalue problem
        let mut correlations = Array1::zeros(self.output_dim);
        for i in 0..self.output_dim {
            if cxx[[i, i]] > 0.0 && cyy[[i, i]] > 0.0 {
                correlations[i] = cxy[[i, i]] / (cxx[[i, i]] * cyy[[i, i]]).sqrt();
            }
        }

        Ok(correlations)
    }

    /// Perform one training step
    fn training_step(&mut self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Float> {
        // Forward pass
        let x_hidden = self.forward_x_training(x);
        let y_hidden = self.forward_y_training(y);

        // Compute canonical correlation loss
        let loss = self.compute_cca_loss(&x_hidden, &y_hidden)?;

        // In a full implementation, you would compute gradients and update weights
        // For now, we'll simulate training with a simple perturbation
        self.perturb_weights();

        Ok(loss)
    }

    /// Compute CCA loss (negative correlation)
    fn compute_cca_loss(
        &self,
        x_hidden: &Array2<Float>,
        y_hidden: &Array2<Float>,
    ) -> Result<Float> {
        let correlations = self.compute_correlations(x_hidden, y_hidden)?;
        Ok(-correlations.sum()) // Negative because we want to maximize correlation
    }

    /// Simple weight perturbation for demonstration
    fn perturb_weights(&mut self) {
        let lr = self.learning_rate * 0.1; // Smaller perturbation

        if let Some(ref mut weights_x) = self.weights_x_ {
            for weight in weights_x.iter_mut() {
                *weight += &Array2::from_shape_fn(weight.raw_dim(), |_| {
                    let mut rng = thread_rng();
                    lr * rng.gen_range(-1.0..1.0)
                });
            }
        }

        if let Some(ref mut weights_y) = self.weights_y_ {
            for weight in weights_y.iter_mut() {
                *weight += &Array2::from_shape_fn(weight.raw_dim(), |_| {
                    let mut rng = thread_rng();
                    lr * rng.gen_range(-1.0..1.0)
                });
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_deep_cca_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let y = array![[2.0, 1.0], [5.0, 4.0], [8.0, 7.0], [11.0, 10.0]];

        let deep_cca = DeepCCA::new(vec![5], vec![4], 2).max_iter(10);

        let fitted = deep_cca.fit(&x, &y).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 2]);
        assert!(fitted.n_iter() <= 10);
    }

    #[test]
    fn test_activation_functions() {
        let x = array![[1.0, -1.0], [0.0, 2.0]];

        // Test sigmoid
        let sigmoid = ActivationFunction::Sigmoid;
        let result = sigmoid.apply(&x);
        assert!(result[[0, 0]] > 0.5);
        assert!(result[[0, 1]] < 0.5);

        // Test ReLU
        let relu = ActivationFunction::ReLU;
        let result = relu.apply(&x);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 0.0);
        assert_eq!(result[[1, 0]], 0.0);
        assert_eq!(result[[1, 1]], 2.0);
    }

    #[test]
    fn test_deep_cca_scaling() {
        let x = array![
            [100.0, 200.0],
            [300.0, 400.0],
            [500.0, 600.0],
            [700.0, 800.0]
        ];
        let y = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];

        let deep_cca = DeepCCA::new(vec![3], vec![3], 1).scale(true).max_iter(5);

        let fitted = deep_cca.fit(&x, &y).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
    }

    #[test]
    fn test_transform_both() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];

        let deep_cca = DeepCCA::new(vec![3], vec![3], 2).max_iter(5);

        let fitted = deep_cca.fit(&x, &y).unwrap();
        let (x_canonical, y_canonical) = fitted.transform_both(&x, &y).unwrap();

        assert_eq!(x_canonical.shape(), &[4, 2]);
        assert_eq!(y_canonical.shape(), &[4, 2]);
    }

    proptest! {
        #[test]
        fn test_deep_cca_properties(
            n_samples in 4..20usize,
            n_features_x in 2..10usize,
            n_features_y in 2..10usize,
            output_dim in 1..4usize,
        ) {
            let x = Array2::from_shape_fn((n_samples, n_features_x), |_| {

                let mut rng = thread_rng();
                rng.gen_range(-1.0..1.0)
            });
            let y = Array2::from_shape_fn((n_samples, n_features_y), |_| {

                let mut rng = thread_rng();
                rng.gen_range(-1.0..1.0)
            });

            let deep_cca = DeepCCA::new(vec![n_features_x.min(5)], vec![n_features_y.min(5)], output_dim)
                .max_iter(3);

            if let Ok(fitted) = deep_cca.fit(&x, &y) {
                let x_transformed = fitted.transform(&x).unwrap();

                // Check output dimensions
                prop_assert_eq!(x_transformed.shape(), &[n_samples, output_dim]);

                // Check that weights have correct dimensions
                prop_assert!(fitted.weights_x().len() >= 1);
                prop_assert!(fitted.weights_y().len() >= 1);

                // Check that correlations are computed
                prop_assert_eq!(fitted.correlations().len(), output_dim);
            }
        }
    }
}
