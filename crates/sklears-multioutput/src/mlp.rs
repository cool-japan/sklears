//! Multi-Layer Perceptron for Multi-Output Learning
//!
//! This module provides a flexible Multi-Layer Perceptron implementation that can handle
//! both regression and classification tasks with multiple outputs. It supports configurable
//! architecture, activation functions, and training parameters.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::RandNormal;
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

use crate::activation::ActivationFunction;
use crate::loss::LossFunction;

/// Multi-Layer Perceptron for Multi-Output Learning
///
/// This neural network can handle both regression and classification tasks with multiple outputs.
/// It supports configurable architecture, activation functions, and training parameters.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::mlp::{MultiOutputMLP};
/// use sklears_multioutput::activation::ActivationFunction;
/// use sklears_multioutput::loss::LossFunction;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let y = array![[0.5, 1.2], [1.0, 2.1], [1.5, 0.8], [2.0, 2.5]]; // Multi-output regression
///
/// let mlp = MultiOutputMLP::new()
///     .hidden_layer_sizes(vec![10, 5])
///     .activation(ActivationFunction::ReLU)
///     .output_activation(ActivationFunction::Linear)
///     .loss_function(LossFunction::MeanSquaredError)
///     .learning_rate(0.01)
///     .max_iter(1000)
///     .random_state(Some(42));
///
/// let trained_mlp = mlp.fit(&X.view(), &y).unwrap();
/// let predictions = trained_mlp.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiOutputMLP<S = Untrained> {
    state: S,
    hidden_layer_sizes: Vec<usize>,
    activation: ActivationFunction,
    output_activation: ActivationFunction,
    loss_function: LossFunction,
    learning_rate: Float,
    max_iter: usize,
    tolerance: Float,
    random_state: Option<u64>,
    alpha: Float, // L2 regularization
    batch_size: Option<usize>,
    early_stopping: bool,
    validation_fraction: Float,
}

/// Trained state for MultiOutputMLP
#[derive(Debug, Clone)]
pub struct MultiOutputMLPTrained {
    /// Weights for each layer
    weights: Vec<Array2<Float>>,
    /// Biases for each layer
    biases: Vec<Array1<Float>>,
    /// Number of input features
    n_features: usize,
    /// Number of outputs
    n_outputs: usize,
    /// Training configuration
    hidden_layer_sizes: Vec<usize>,
    activation: ActivationFunction,
    output_activation: ActivationFunction,
    /// Training history
    loss_curve: Vec<Float>,
    /// Number of iterations performed
    n_iter: usize,
}

impl MultiOutputMLP<Untrained> {
    /// Create a new MultiOutputMLP instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            hidden_layer_sizes: vec![100],
            activation: ActivationFunction::ReLU,
            output_activation: ActivationFunction::Linear,
            loss_function: LossFunction::MeanSquaredError,
            learning_rate: 0.001,
            max_iter: 200,
            tolerance: 1e-4,
            random_state: None,
            alpha: 0.0001,
            batch_size: None,
            early_stopping: false,
            validation_fraction: 0.1,
        }
    }

    /// Set hidden layer sizes
    pub fn hidden_layer_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_layer_sizes = sizes;
        self
    }

    /// Set activation function for hidden layers
    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        self.activation = activation;
        self
    }

    /// Set activation function for output layer
    pub fn output_activation(mut self, activation: ActivationFunction) -> Self {
        self.output_activation = activation;
        self
    }

    /// Set loss function
    pub fn loss_function(mut self, loss_function: LossFunction) -> Self {
        self.loss_function = loss_function;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set L2 regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set batch size for mini-batch gradient descent
    pub fn batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable early stopping
    pub fn early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    /// Set validation fraction for early stopping
    pub fn validation_fraction(mut self, validation_fraction: Float) -> Self {
        self.validation_fraction = validation_fraction;
        self
    }
}

impl Default for MultiOutputMLP<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiOutputMLP<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<Float>> for MultiOutputMLP<Untrained> {
    type Fitted = MultiOutputMLP<MultiOutputMLPTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let (n_samples_y, n_outputs) = y.dim();

        if n_samples != n_samples_y {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit with zero samples".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = match self.random_state {
            Some(seed) => scirs2_core::random::seeded_rng(seed),
            None => scirs2_core::random::seeded_rng(42),
        };

        // Build network architecture
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend(&self.hidden_layer_sizes);
        layer_sizes.push(n_outputs);

        // Initialize weights and biases
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Xavier/Glorot initialization
            let scale = (2.0 / (input_size + output_size) as Float).sqrt();
            let normal_dist = RandNormal::new(0.0, scale).unwrap();
            let mut weight_matrix = Array2::<Float>::zeros((output_size, input_size));
            for i in 0..output_size {
                for j in 0..input_size {
                    weight_matrix[[i, j]] = rng.sample(normal_dist);
                }
            }
            let bias_vector = Array1::<Float>::zeros(output_size);

            weights.push(weight_matrix);
            biases.push(bias_vector);
        }

        // Training loop
        let mut loss_curve = Vec::new();
        let X_owned = X.to_owned();
        let y_owned = y.to_owned();

        for epoch in 0..self.max_iter {
            // Forward pass
            let (activations, _) = self.forward_pass(&X_owned, &weights, &biases)?;
            let predictions = activations.last().unwrap();

            // Compute loss
            let loss = self.loss_function.compute_loss(predictions, &y_owned);
            loss_curve.push(loss);

            // Check convergence
            if epoch > 0 && (loss_curve[epoch - 1] - loss).abs() < self.tolerance {
                break;
            }

            // Backward pass
            self.backward_pass(&X_owned, &y_owned, &mut weights, &mut biases)?;
        }

        let trained_state = MultiOutputMLPTrained {
            weights,
            biases,
            n_features,
            n_outputs,
            hidden_layer_sizes: self.hidden_layer_sizes.clone(),
            activation: self.activation,
            output_activation: self.output_activation,
            loss_curve,
            n_iter: self.max_iter,
        };

        Ok(MultiOutputMLP {
            state: trained_state,
            hidden_layer_sizes: self.hidden_layer_sizes,
            activation: self.activation,
            output_activation: self.output_activation,
            loss_function: self.loss_function,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            random_state: self.random_state,
            alpha: self.alpha,
            batch_size: self.batch_size,
            early_stopping: self.early_stopping,
            validation_fraction: self.validation_fraction,
        })
    }
}

impl MultiOutputMLP<Untrained> {
    /// Forward pass through the network
    fn forward_pass(
        &self,
        X: &Array2<Float>,
        weights: &[Array2<Float>],
        biases: &[Array1<Float>],
    ) -> SklResult<(Vec<Array2<Float>>, Vec<Array2<Float>>)> {
        let mut activations = vec![X.clone()];
        let mut z_values = Vec::new();

        for (i, (weight, bias)) in weights.iter().zip(biases.iter()).enumerate() {
            let current_input = activations.last().unwrap();

            // Linear transformation: z = X * W^T + b
            let z = current_input.dot(&weight.t()) + bias.view().insert_axis(Axis(0));
            z_values.push(z.clone());

            // Apply activation function
            let activation_fn = if i == weights.len() - 1 {
                self.output_activation
            } else {
                self.activation
            };

            let activated = activation_fn.apply_2d(&z);
            activations.push(activated);
        }

        Ok((activations, z_values))
    }

    /// Backward pass with gradient computation
    fn backward_pass(
        &self,
        X: &Array2<Float>,
        y: &Array2<Float>,
        weights: &mut [Array2<Float>],
        biases: &mut [Array1<Float>],
    ) -> SklResult<()> {
        let (activations, z_values) = self.forward_pass(X, weights, biases)?;
        let n_samples = X.nrows() as Float;

        // Compute output layer error
        let output_predictions = activations.last().unwrap();
        let mut delta = output_predictions - y;

        // Backpropagate errors
        for i in (0..weights.len()).rev() {
            let current_activation = &activations[i];

            // Compute gradients
            let weight_gradient = delta.t().dot(current_activation) / n_samples;
            let bias_gradient = delta.mean_axis(Axis(0)).unwrap();

            // Add L2 regularization to weight gradient
            let regularized_weight_gradient = weight_gradient + self.alpha * &weights[i];

            // Update weights and biases
            weights[i] = &weights[i] - self.learning_rate * regularized_weight_gradient;
            biases[i] = &biases[i] - self.learning_rate * bias_gradient;

            // Compute delta for next layer (if not the first layer)
            if i > 0 {
                let activation_fn = if i == weights.len() - 1 {
                    self.output_activation
                } else {
                    self.activation
                };

                // For simplicity, we'll use a basic derivative approximation
                let derivative_approx = match activation_fn {
                    ActivationFunction::ReLU => {
                        z_values[i - 1].map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
                    }
                    ActivationFunction::Sigmoid => {
                        let sigmoid_vals = &activations[i];
                        sigmoid_vals.map(|&val| val * (1.0 - val))
                    }
                    ActivationFunction::Tanh => {
                        let tanh_vals = &activations[i];
                        tanh_vals.map(|&val| 1.0 - val * val)
                    }
                    _ => Array2::ones(z_values[i - 1].dim()),
                };

                delta = delta.dot(&weights[i]) * derivative_approx;
            }
        }

        Ok(())
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>> for MultiOutputMLP<MultiOutputMLPTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let X_owned = X.to_owned();
        let (activations, _) = self.forward_pass_trained(&X_owned)?;
        let predictions = activations.last().unwrap().clone();

        Ok(predictions)
    }
}

impl MultiOutputMLP<MultiOutputMLPTrained> {
    /// Forward pass for trained model
    fn forward_pass_trained(
        &self,
        X: &Array2<Float>,
    ) -> SklResult<(Vec<Array2<Float>>, Vec<Array2<Float>>)> {
        let mut activations = vec![X.clone()];
        let mut z_values = Vec::new();

        for (i, (weight, bias)) in self
            .state
            .weights
            .iter()
            .zip(self.state.biases.iter())
            .enumerate()
        {
            let current_input = activations.last().unwrap();

            // Linear transformation: z = X * W^T + b
            let z = current_input.dot(&weight.t()) + bias.view().insert_axis(Axis(0));
            z_values.push(z.clone());

            // Apply activation function
            let activation_fn = if i == self.state.weights.len() - 1 {
                self.state.output_activation
            } else {
                self.state.activation
            };

            let activated = activation_fn.apply_2d(&z);
            activations.push(activated);
        }

        Ok((activations, z_values))
    }

    /// Get the loss curve from training
    pub fn loss_curve(&self) -> &[Float] {
        &self.state.loss_curve
    }

    /// Get the number of iterations performed during training
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the network weights
    pub fn weights(&self) -> &[Array2<Float>] {
        &self.state.weights
    }

    /// Get the network biases
    pub fn biases(&self) -> &[Array1<Float>] {
        &self.state.biases
    }
}

/// Multi-Output MLP Classifier
///
/// This is a specialized version of MultiOutputMLP for classification tasks.
/// It automatically configures the network for multi-class or multi-label classification.
pub type MultiOutputMLPClassifier<S = Untrained> = MultiOutputMLP<S>;

impl MultiOutputMLPClassifier<Untrained> {
    /// Create a new classifier with appropriate defaults
    pub fn new_classifier() -> Self {
        Self::new()
            .output_activation(ActivationFunction::Sigmoid)
            .loss_function(LossFunction::BinaryCrossEntropy)
    }
}

/// Multi-Output MLP Regressor
///
/// This is a specialized version of MultiOutputMLP for regression tasks.
/// It automatically configures the network for multi-output regression.
pub type MultiOutputMLPRegressor<S = Untrained> = MultiOutputMLP<S>;

impl MultiOutputMLPRegressor<Untrained> {
    /// Create a new regressor with appropriate defaults
    pub fn new_regressor() -> Self {
        Self::new()
            .output_activation(ActivationFunction::Linear)
            .loss_function(LossFunction::MeanSquaredError)
    }
}
