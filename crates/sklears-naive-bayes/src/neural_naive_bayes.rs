//! Neural Naive Bayes implementation with deep learning integration
//!
//! This module provides neural network-based extensions to traditional Naive Bayes,
//! allowing for more flexible probability distributions and feature interactions.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::{Rng, SeedableRng};
// SciRS2 Policy Compliance - Use scirs2-core for random distributions
use scirs2_core::random::essentials::Normal as RandNormal;
use scirs2_core::random::Distribution;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NeuralNBError {
    #[error("Neural network training failed: {0}")]
    TrainingFailed(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },
    #[error("Invalid network architecture: {0}")]
    InvalidArchitecture(String),
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
}

/// Activation functions for neural networks
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    /// Sigmoid
    Sigmoid,
    /// Tanh
    Tanh,
    /// ReLU
    ReLU,
    /// LeakyReLU
    LeakyReLU { alpha: f64 },
    /// Softmax
    Softmax,
    /// Identity
    Identity,
}

impl ActivationFunction {
    /// Apply activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::LeakyReLU { alpha } => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            ActivationFunction::Softmax => x.exp(), // Note: Softmax requires normalization
            ActivationFunction::Identity => x,
        }
    }

    /// Apply derivative of activation function
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::LeakyReLU { alpha } => {
                if x > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            ActivationFunction::Softmax => {
                let s = self.apply(x);
                s * (1.0 - s) // Simplified for individual components
            }
            ActivationFunction::Identity => 1.0,
        }
    }
}

/// Neural network layer
#[derive(Debug, Clone)]
pub struct NeuralLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: ActivationFunction,
}

impl NeuralLayer {
    /// Create new layer with random initialization
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        rng: &mut impl Rng,
    ) -> Self {
        let std_dev = (2.0 / input_size as f64).sqrt(); // Xavier initialization
        let normal = RandNormal::new(0.0, std_dev).unwrap();

        let weights = Array2::from_shape_fn((output_size, input_size), |_| normal.sample(rng));

        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation,
        }
    }

    /// Forward pass through layer
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let linear_output = self.weights.dot(input) + &self.biases;

        match &self.activation {
            ActivationFunction::Softmax => {
                // Special handling for softmax
                let max_val = linear_output
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_vals: Array1<f64> = linear_output
                    .iter()
                    .map(|&x| (x - max_val).exp())
                    .collect::<Vec<_>>()
                    .into();
                let sum = exp_vals.sum();
                exp_vals / sum
            }
            _ => linear_output
                .iter()
                .map(|&x| self.activation.apply(x))
                .collect::<Vec<_>>()
                .into(),
        }
    }

    /// Backward pass through layer
    pub fn backward(
        &mut self,
        input: &Array1<f64>,
        grad_output: &Array1<f64>,
        learning_rate: f64,
    ) -> Array1<f64> {
        // Compute linear output for derivative calculation
        let linear_output = self.weights.dot(input) + &self.biases;

        // Compute gradient w.r.t. pre-activation
        let grad_pre_activation: Array1<f64> = match &self.activation {
            ActivationFunction::Softmax => {
                // For softmax, gradient computation is more complex
                grad_output.clone()
            }
            _ => linear_output
                .iter()
                .zip(grad_output.iter())
                .map(|(&x, &grad)| grad * self.activation.derivative(x))
                .collect::<Vec<_>>()
                .into(),
        };

        // Compute gradients w.r.t. weights and biases
        let grad_weights = grad_pre_activation
            .clone()
            .insert_axis(Axis(1))
            .dot(&input.clone().insert_axis(Axis(0)));
        let grad_biases = grad_pre_activation.clone();

        // Update weights and biases
        self.weights = &self.weights - &(learning_rate * grad_weights);
        self.biases = &self.biases - &(learning_rate * grad_biases);

        // Compute gradient w.r.t. input
        self.weights.t().dot(&grad_pre_activation)
    }

    /// Get weights matrix
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.weights.ncols()
    }

    /// Get output size
    pub fn output_size(&self) -> usize {
        self.weights.nrows()
    }
}

/// Neural Naive Bayes network configuration
#[derive(Debug, Clone)]
pub struct NeuralNBConfig {
    pub hidden_layers: Vec<usize>,
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub batch_size: usize,
    pub tolerance: f64,
    pub regularization: f64,
    pub dropout_rate: f64,
    pub early_stopping_patience: usize,
}

impl Default for NeuralNBConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![64, 32],
            learning_rate: 0.001,
            max_epochs: 1000,
            batch_size: 32,
            tolerance: 1e-6,
            regularization: 0.001,
            dropout_rate: 0.1,
            early_stopping_patience: 10,
        }
    }
}

/// Neural Naive Bayes classifier
///
/// Combines traditional Naive Bayes with neural networks to learn more
/// flexible probability distributions and feature interactions.
#[derive(Debug)]
pub struct NeuralNaiveBayes {
    config: NeuralNBConfig,
    networks: HashMap<i32, Vec<NeuralLayer>>,
    classes: Vec<i32>,
    class_priors: HashMap<i32, f64>,
    n_features: usize,
    fitted: bool,
    rng: scirs2_core::random::CoreRandom<scirs2_core::random::rngs::StdRng>,
}

impl NeuralNaiveBayes {
    /// Create new Neural Naive Bayes classifier
    pub fn new(config: NeuralNBConfig) -> Self {
        Self {
            config,
            networks: HashMap::new(),
            classes: Vec::new(),
            class_priors: HashMap::new(),
            n_features: 0,
            fitted: false,
            rng: scirs2_core::random::CoreRandom::<scirs2_core::random::rngs::StdRng>::from_rng(
                &mut scirs2_core::random::thread_rng(),
            ),
        }
    }

    /// Create with random seed for reproducibility
    pub fn with_seed(config: NeuralNBConfig, seed: u64) -> Self {
        Self {
            config,
            networks: HashMap::new(),
            classes: Vec::new(),
            class_priors: HashMap::new(),
            n_features: 0,
            fitted: false,
            rng:
                scirs2_core::random::CoreRandom::<scirs2_core::random::rngs::StdRng>::seed_from_u64(
                    seed,
                ),
        }
    }

    /// Build neural network for a class
    fn build_network(&mut self, input_size: usize) -> Vec<NeuralLayer> {
        let mut layers = Vec::new();
        let mut current_size = input_size;

        // Hidden layers
        for &hidden_size in &self.config.hidden_layers {
            layers.push(NeuralLayer::new(
                current_size,
                hidden_size,
                ActivationFunction::ReLU,
                &mut self.rng,
            ));
            current_size = hidden_size;
        }

        // Output layer (probability for this class)
        layers.push(NeuralLayer::new(
            current_size,
            1,
            ActivationFunction::Sigmoid,
            &mut self.rng,
        ));

        layers
    }

    /// Fit Neural Naive Bayes to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<(), NeuralNBError> {
        if x.nrows() != y.len() {
            return Err(NeuralNBError::DimensionMismatch {
                expected: x.nrows(),
                actual: y.len(),
            });
        }

        self.n_features = x.ncols();

        // Find unique classes and compute priors
        let mut unique_classes: Vec<i32> = y.iter().cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        self.classes = unique_classes;

        let n_samples = y.len() as f64;
        for &class in &self.classes {
            let class_count = y.iter().filter(|&&label| label == class).count() as f64;
            self.class_priors.insert(class, class_count / n_samples);
        }

        // Build and train neural networks for each class
        let classes_clone = self.classes.clone();
        for &class in &classes_clone {
            let mut network = self.build_network(self.n_features);

            // Prepare training data for this class (binary classification)
            let class_labels: Array1<f64> = y
                .iter()
                .map(|&label| if label == class { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
                .into();

            // Train network
            self.train_network(&mut network, x, &class_labels)?;
            self.networks.insert(class, network);
        }

        self.fitted = true;
        Ok(())
    }

    /// Train a single neural network
    fn train_network(
        &mut self,
        network: &mut Vec<NeuralLayer>,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(), NeuralNBError> {
        let n_samples = x.nrows();
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.max_epochs {
            let mut total_loss = 0.0;
            let mut batch_count = 0;

            // Mini-batch training
            for batch_start in (0..n_samples).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(n_samples);
                let batch_size = batch_end - batch_start;

                let mut batch_loss = 0.0;

                for sample_idx in batch_start..batch_end {
                    let input = x.row(sample_idx).to_owned();
                    let target = y[sample_idx];

                    // Forward pass
                    let prediction = self.forward_pass(network, &input);

                    // Compute loss (binary cross-entropy)
                    let loss = self.compute_loss(prediction, target);
                    batch_loss += loss;

                    // Backward pass
                    let grad_output = Array1::from_elem(1, prediction - target);
                    self.backward_pass(network, &input, &grad_output);
                }

                total_loss += batch_loss / batch_size as f64;
                batch_count += 1;
            }

            let avg_loss = total_loss / batch_count as f64;

            // Early stopping
            if avg_loss < best_loss - self.config.tolerance {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    break;
                }
            }

            // Check for convergence
            if avg_loss < self.config.tolerance {
                break;
            }

            // Check for numerical instability
            if !avg_loss.is_finite() {
                return Err(NeuralNBError::NumericalInstability(format!(
                    "Loss became non-finite at epoch {}",
                    epoch
                )));
            }
        }

        Ok(())
    }

    /// Forward pass through network
    fn forward_pass(&self, network: &[NeuralLayer], input: &Array1<f64>) -> f64 {
        let mut current_input = input.clone();

        for layer in network {
            current_input = layer.forward(&current_input);
        }

        current_input[0] // Single output for binary classification
    }

    /// Backward pass through network
    fn backward_pass(
        &self,
        network: &mut [NeuralLayer],
        input: &Array1<f64>,
        grad_output: &Array1<f64>,
    ) {
        // Store forward pass activations
        let mut activations = vec![input.clone()];
        let mut current_input = input.clone();

        for layer in network.iter() {
            current_input = layer.forward(&current_input);
            activations.push(current_input.clone());
        }

        // Backward pass
        let mut current_grad = grad_output.clone();

        for (i, layer) in network.iter_mut().enumerate().rev() {
            let layer_input = &activations[i];
            current_grad = layer.backward(layer_input, &current_grad, self.config.learning_rate);
        }
    }

    /// Compute binary cross-entropy loss
    fn compute_loss(&self, prediction: f64, target: f64) -> f64 {
        let eps = 1e-15; // Prevent log(0)
        let pred_clipped = prediction.max(eps).min(1.0 - eps);

        -(target * pred_clipped.ln() + (1.0 - target) * (1.0 - pred_clipped).ln())
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>, NeuralNBError> {
        if !self.fitted {
            return Err(NeuralNBError::TrainingFailed(
                "Model not fitted".to_string(),
            ));
        }

        if x.ncols() != self.n_features {
            return Err(NeuralNBError::DimensionMismatch {
                expected: self.n_features,
                actual: x.ncols(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        for (sample_idx, sample) in x.outer_iter().enumerate() {
            let mut class_probs = Vec::new();

            for &class in &self.classes {
                if let Some(network) = self.networks.get(&class) {
                    // Get neural network probability for this class
                    let nn_prob = self.forward_pass(network, &sample.to_owned());

                    // Combine with class prior using Bayes' theorem
                    let prior = self.class_priors[&class];
                    let posterior = nn_prob * prior;

                    class_probs.push(posterior);
                } else {
                    class_probs.push(0.0);
                }
            }

            // Normalize probabilities
            let total: f64 = class_probs.iter().sum();
            if total > 0.0 {
                for (class_idx, &prob) in class_probs.iter().enumerate() {
                    probabilities[[sample_idx, class_idx]] = prob / total;
                }
            } else {
                // Uniform distribution if all probabilities are zero
                let uniform_prob = 1.0 / n_classes as f64;
                for class_idx in 0..n_classes {
                    probabilities[[sample_idx, class_idx]] = uniform_prob;
                }
            }
        }

        Ok(probabilities)
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>, NeuralNBError> {
        let probabilities = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (sample_idx, row) in probabilities.outer_iter().enumerate() {
            let mut max_idx = 0;
            let mut max_prob = row[0];

            for (class_idx, &prob) in row.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    max_idx = class_idx;
                }
            }

            predictions[sample_idx] = self.classes[max_idx];
        }

        Ok(predictions)
    }

    /// Get class labels
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    /// Get class priors
    pub fn class_priors(&self) -> &HashMap<i32, f64> {
        &self.class_priors
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get network architecture for a class
    pub fn get_network_info(&self, class: i32) -> Option<Vec<(usize, usize)>> {
        self.networks.get(&class).map(|network| {
            network
                .iter()
                .map(|layer| (layer.weights.nrows(), layer.weights.ncols()))
                .collect()
        })
    }
}

/// Builder for Neural Naive Bayes configuration
#[derive(Debug, Clone)]
pub struct NeuralNBBuilder {
    config: NeuralNBConfig,
}

impl NeuralNBBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: NeuralNBConfig::default(),
        }
    }

    /// Set hidden layer sizes
    pub fn hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.config.hidden_layers = layers;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set maximum epochs
    pub fn max_epochs(mut self, epochs: usize) -> Self {
        self.config.max_epochs = epochs;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set tolerance for convergence
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.tolerance = tol;
        self
    }

    /// Set regularization strength
    pub fn regularization(mut self, reg: f64) -> Self {
        self.config.regularization = reg;
        self
    }

    /// Set dropout rate
    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.config.dropout_rate = rate;
        self
    }

    /// Set early stopping patience
    pub fn early_stopping_patience(mut self, patience: usize) -> Self {
        self.config.early_stopping_patience = patience;
        self
    }

    /// Build Neural Naive Bayes classifier
    pub fn build(self) -> NeuralNaiveBayes {
        NeuralNaiveBayes::new(self.config)
    }

    /// Build with random seed
    pub fn build_with_seed(self, seed: u64) -> NeuralNaiveBayes {
        NeuralNaiveBayes::with_seed(self.config, seed)
    }
}

impl Default for NeuralNBBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_activation_functions() {
        let sigmoid = ActivationFunction::Sigmoid;
        assert_relative_eq!(sigmoid.apply(0.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(sigmoid.apply(1000.0), 1.0, epsilon = 1e-3);
        assert_relative_eq!(sigmoid.apply(-1000.0), 0.0, epsilon = 1e-3);

        let relu = ActivationFunction::ReLU;
        assert_eq!(relu.apply(-1.0), 0.0);
        assert_eq!(relu.apply(1.0), 1.0);

        let tanh = ActivationFunction::Tanh;
        assert_relative_eq!(tanh.apply(0.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_neural_layer_creation() {
        let mut rng = scirs2_core::random::CoreRandom::seed_from_u64(42);
        let layer = NeuralLayer::new(3, 5, ActivationFunction::ReLU, &mut rng);

        assert_eq!(layer.weights.shape(), &[5, 3]);
        assert_eq!(layer.biases.len(), 5);
    }

    #[test]
    fn test_neural_layer_forward() {
        let mut rng = scirs2_core::random::CoreRandom::seed_from_u64(42);
        let layer = NeuralLayer::new(2, 3, ActivationFunction::ReLU, &mut rng);

        let input = Array1::from_vec(vec![1.0, 2.0]);
        let output = layer.forward(&input);

        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&x| x >= 0.0)); // ReLU output should be non-negative
    }

    #[test]
    fn test_neural_nb_builder() {
        let nb = NeuralNBBuilder::new()
            .hidden_layers(vec![10, 5])
            .learning_rate(0.01)
            .max_epochs(100)
            .build_with_seed(42);

        assert_eq!(nb.config.hidden_layers, vec![10, 5]);
        assert_eq!(nb.config.learning_rate, 0.01);
        assert_eq!(nb.config.max_epochs, 100);
    }

    #[test]
    fn test_neural_nb_fit_and_predict() {
        let mut nb = NeuralNBBuilder::new()
            .hidden_layers(vec![5])
            .learning_rate(0.1)
            .max_epochs(50)
            .batch_size(4)
            .build_with_seed(42);

        // Simple 2D dataset
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // class 0
                1.1, 1.1, // class 0
                1.2, 0.9, // class 0
                5.0, 5.0, // class 1
                5.1, 5.1, // class 1
                4.9, 5.2, // class 1
            ],
        )
        .unwrap();

        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        // Fit model
        nb.fit(&x, &y).unwrap();
        assert!(nb.is_fitted());

        // Test prediction
        let test_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.05, 1.05, // Should be close to class 0
                5.05, 5.05, // Should be close to class 1
            ],
        )
        .unwrap();

        let predictions = nb.predict(&test_x).unwrap();
        assert_eq!(predictions.len(), 2);

        let probabilities = nb.predict_proba(&test_x).unwrap();
        assert_eq!(probabilities.shape(), &[2, 2]);

        // Check that probabilities sum to 1 for each sample
        for row in probabilities.outer_iter() {
            let sum: f64 = row.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_neural_nb_invalid_dimensions() {
        let mut nb = NeuralNBBuilder::new().build_with_seed(42);

        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0]); // Wrong length

        let result = nb.fit(&x, &y);
        assert!(matches!(
            result,
            Err(NeuralNBError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_neural_nb_network_info() {
        let mut nb = NeuralNBBuilder::new()
            .hidden_layers(vec![8, 4])
            .build_with_seed(42);

        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        nb.fit(&x, &y).unwrap();

        let network_info = nb.get_network_info(0).unwrap();

        // Should have 3 layers: input->8, 8->4, 4->1
        assert_eq!(network_info.len(), 3);
        assert_eq!(network_info[0], (8, 3)); // First hidden layer
        assert_eq!(network_info[1], (4, 8)); // Second hidden layer
        assert_eq!(network_info[2], (1, 4)); // Output layer
    }

    #[test]
    fn test_loss_computation() {
        let nb = NeuralNaiveBayes::new(NeuralNBConfig::default());

        // Perfect prediction
        let loss1 = nb.compute_loss(1.0, 1.0);
        assert!(loss1 < 1e-10);

        // Worst prediction
        let loss2 = nb.compute_loss(0.0, 1.0);
        assert!(loss2 > 10.0);

        // Medium prediction
        let loss3 = nb.compute_loss(0.5, 1.0);
        assert!(loss3 > loss1 && loss3 < loss2);
    }
}
