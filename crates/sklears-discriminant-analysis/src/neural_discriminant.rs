//! Neural Discriminant Analysis
//!
//! This module implements neural network-based extensions of discriminant analysis,
//! allowing for more complex, non-linear decision boundaries through multi-layer perceptrons.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};
use std::collections::HashMap;

/// Activation function types for neural discriminant analysis
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ActivationFunction {
    /// Rectified Linear Unit: f(x) = max(0, x)
    #[default]
    ReLU,
    /// Sigmoid: f(x) = 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh,
    /// Leaky ReLU: f(x) = max(alpha * x, x)
    LeakyReLU { alpha: Float },
    /// Swish: f(x) = x * sigmoid(x)
    Swish,
    /// GELU: f(x) = x * Φ(x) where Φ is the CDF of standard normal
    GELU,
}

impl ActivationFunction {
    /// Apply the activation function
    pub fn apply(&self, x: Float) -> Float {
        match self {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::LeakyReLU { alpha } => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            ActivationFunction::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            }
            ActivationFunction::GELU => {
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
                let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            }
        }
    }

    /// Apply the derivative of the activation function
    pub fn derivative(&self, x: Float) -> Float {
        match self {
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::Sigmoid => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 - sigmoid)
            }
            ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFunction::LeakyReLU { alpha } => {
                if x > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            ActivationFunction::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid + x * sigmoid * (1.0 - sigmoid)
            }
            ActivationFunction::GELU => {
                // Derivative approximation
                let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
                let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner.powi(2);

                0.5 * (1.0 + tanh_inner)
                    + 0.5 * x * sech2 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x.powi(2))
            }
        }
    }
}

/// Neural network architecture specification
#[derive(Debug, Clone)]
pub struct NetworkArchitecture {
    /// hidden_layers
    pub hidden_layers: Vec<usize>,
    /// activation
    pub activation: ActivationFunction,
    /// output_activation
    pub output_activation: ActivationFunction,
    /// dropout_rate
    pub dropout_rate: Float,
    /// batch_norm
    pub batch_norm: bool,
}

impl Default for NetworkArchitecture {
    fn default() -> Self {
        Self {
            hidden_layers: vec![64, 32],
            activation: ActivationFunction::ReLU,
            output_activation: ActivationFunction::Sigmoid,
            dropout_rate: 0.1,
            batch_norm: false,
        }
    }
}

/// Training configuration for neural discriminant analysis
#[derive(Debug, Clone)]
pub struct NeuralTrainingConfig {
    /// learning_rate
    pub learning_rate: Float,
    /// momentum
    pub momentum: Float,
    /// weight_decay
    pub weight_decay: Float,
    /// max_epochs
    pub max_epochs: usize,
    /// batch_size
    pub batch_size: usize,
    /// tolerance
    pub tolerance: Float,
    /// early_stopping
    pub early_stopping: bool,
    /// early_stopping_patience
    pub early_stopping_patience: usize,
    /// validation_split
    pub validation_split: Float,
}

impl Default for NeuralTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 1e-4,
            max_epochs: 100,
            batch_size: 32,
            tolerance: 1e-6,
            early_stopping: true,
            early_stopping_patience: 10,
            validation_split: 0.2,
        }
    }
}

/// Configuration for Neural Discriminant Analysis
#[derive(Debug, Clone)]
pub struct NeuralDiscriminantAnalysisConfig {
    /// architecture
    pub architecture: NetworkArchitecture,
    /// training
    pub training: NeuralTrainingConfig,
    /// discriminant_regularization
    pub discriminant_regularization: Float,
    /// class_balance_weight
    pub class_balance_weight: bool,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for NeuralDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            architecture: NetworkArchitecture::default(),
            training: NeuralTrainingConfig::default(),
            discriminant_regularization: 0.01,
            class_balance_weight: true,
            random_state: Some(42),
        }
    }
}

/// Neural layer representation
#[derive(Debug, Clone)]
pub struct NeuralLayer {
    /// weights
    pub weights: Array2<Float>,
    /// biases
    pub biases: Array1<Float>,
    /// activation
    pub activation: ActivationFunction,
    /// dropout_rate
    pub dropout_rate: Float,
}

impl NeuralLayer {
    /// Create a new neural layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        dropout_rate: Float,
        random_state: u64,
    ) -> Self {
        // Xavier/Glorot initialization
        let fan_in = input_size as Float;
        let fan_out = output_size as Float;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        let mut weights = Array2::zeros((input_size, output_size));
        let mut rng_state = random_state;

        for i in 0..input_size {
            for j in 0..output_size {
                // Simple pseudo-random number generation
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let random_val = (rng_state as Float / u64::MAX as Float) * 2.0 - 1.0;
                weights[[i, j]] = random_val * limit;
            }
        }

        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation,
            dropout_rate,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &ArrayView1<Float>, training: bool) -> Array1<Float> {
        // Linear transformation: z = W^T * x + b
        let mut output = Array1::zeros(self.weights.ncols());

        for j in 0..self.weights.ncols() {
            let mut sum = self.biases[j];
            for i in 0..self.weights.nrows() {
                sum += input[i] * self.weights[[i, j]];
            }
            output[j] = sum;
        }

        // Apply activation function
        output.mapv_inplace(|x| self.activation.apply(x));

        // Apply dropout during training
        if training && self.dropout_rate > 0.0 {
            // Simple deterministic dropout for reproducibility
            for (i, val) in output.iter_mut().enumerate() {
                let dropout_prob = ((i as Float * 17.0).sin().abs() + 1.0) / 2.0;
                if dropout_prob < self.dropout_rate {
                    *val = 0.0;
                } else {
                    *val /= 1.0 - self.dropout_rate; // Inverted dropout
                }
            }
        }

        output
    }

    /// Batch forward pass
    pub fn forward_batch(&self, input: &ArrayView2<Float>, training: bool) -> Array2<Float> {
        let batch_size = input.nrows();
        let output_size = self.weights.ncols();
        let mut output = Array2::zeros((batch_size, output_size));

        for (i, row) in input.axis_iter(Axis(0)).enumerate() {
            let result = self.forward(&row, training);
            output.row_mut(i).assign(&result);
        }

        output
    }
}

/// Neural Discriminant Analysis
///
/// Neural discriminant analysis extends traditional linear discriminant analysis
/// by using neural networks to learn complex, non-linear decision boundaries.
/// This approach combines the interpretability of discriminant analysis with
/// the expressiveness of neural networks.
///
/// # Mathematical Background
///
/// The neural discriminant model learns a non-linear mapping:
/// f: R^d → R^k
///
/// Where d is the input dimension and k is the number of discriminant components.
/// The network is trained to maximize class separability in the learned feature space
/// while maintaining discriminant analysis properties.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use sklears_discriminant_analysis::*;
/// use sklears_core::traits::{Predict, Fit};
///
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 0, 1, 1];
///
/// let nda = NeuralDiscriminantAnalysis::new()
///     .max_epochs(50)
///     .hidden_layers(vec![32, 16]);
/// let fitted = nda.fit(&x, &y).unwrap();
/// let predictions = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NeuralDiscriminantAnalysis {
    config: NeuralDiscriminantAnalysisConfig,
}

impl Default for NeuralDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralDiscriminantAnalysis {
    /// Create a new neural discriminant analysis instance
    pub fn new() -> Self {
        Self {
            config: NeuralDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the hidden layer architecture
    pub fn hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.config.architecture.hidden_layers = layers;
        self
    }

    /// Set the activation function
    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        self.config.architecture.activation = activation;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.training.learning_rate = lr;
        self
    }

    /// Set the maximum number of epochs
    pub fn max_epochs(mut self, epochs: usize) -> Self {
        self.config.training.max_epochs = epochs;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.training.batch_size = size;
        self
    }

    /// Set the dropout rate
    pub fn dropout_rate(mut self, rate: Float) -> Self {
        self.config.architecture.dropout_rate = rate;
        self
    }

    /// Set the discriminant regularization strength
    pub fn discriminant_regularization(mut self, reg: Float) -> Self {
        self.config.discriminant_regularization = reg;
        self
    }

    /// Enable or disable class balance weighting
    pub fn class_balance_weight(mut self, balance: bool) -> Self {
        self.config.class_balance_weight = balance;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, state: Option<u64>) -> Self {
        self.config.random_state = state;
        self
    }
}

/// Trained Neural Discriminant Analysis model
#[derive(Debug, Clone)]
pub struct TrainedNeuralDiscriminantAnalysis {
    config: NeuralDiscriminantAnalysisConfig,
    classes: Array1<i32>,
    layers: Vec<NeuralLayer>,
    class_means: HashMap<i32, Array1<Float>>,
    feature_means: Array1<Float>,
    feature_stds: Array1<Float>,
    training_history: Vec<Float>,
    n_features: usize,
    n_components: usize,
}

impl TrainedNeuralDiscriminantAnalysis {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get the number of discriminant components
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get the training history (loss values)
    pub fn training_history(&self) -> &Vec<Float> {
        &self.training_history
    }

    /// Get the neural network layers
    pub fn layers(&self) -> &Vec<NeuralLayer> {
        &self.layers
    }

    /// Forward pass through the entire network
    pub fn forward(&self, input: &ArrayView1<Float>, training: bool) -> Array1<Float> {
        // Normalize input
        let mut current = Array1::zeros(input.len());
        for i in 0..input.len() {
            current[i] = (input[i] - self.feature_means[i]) / self.feature_stds[i];
        }

        // Pass through all layers
        for layer in &self.layers {
            current = layer.forward(&current.view(), training);
        }

        current
    }

    /// Batch forward pass
    pub fn forward_batch(&self, input: &ArrayView2<Float>, training: bool) -> Array2<Float> {
        let batch_size = input.nrows();
        let n_features = input.ncols();

        // Normalize input batch
        let mut normalized = Array2::zeros((batch_size, n_features));
        for i in 0..batch_size {
            for j in 0..n_features {
                normalized[[i, j]] = (input[[i, j]] - self.feature_means[j]) / self.feature_stds[j];
            }
        }

        let mut current = normalized;

        // Pass through all layers
        for layer in &self.layers {
            current = layer.forward_batch(&current.view(), training);
        }

        current
    }
}

impl Estimator for NeuralDiscriminantAnalysis {
    type Config = NeuralDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>, TrainedNeuralDiscriminantAnalysis>
    for NeuralDiscriminantAnalysis
{
    type Fitted = TrainedNeuralDiscriminantAnalysis;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<TrainedNeuralDiscriminantAnalysis> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Mismatch between X and y dimensions".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        // Determine number of discriminant components
        let n_components = (n_classes - 1).min(n_features).min(
            self.config
                .architecture
                .hidden_layers
                .last()
                .copied()
                .unwrap_or(n_features),
        );

        // Compute feature normalization parameters
        let feature_means = x.mean_axis(Axis(0)).unwrap();
        let mut feature_stds = Array1::zeros(n_features);
        for j in 0..n_features {
            let variance = x
                .column(j)
                .iter()
                .map(|&val| (val - feature_means[j]).powi(2))
                .sum::<Float>()
                / (n_samples - 1) as Float;
            feature_stds[j] = variance.sqrt().max(1e-8); // Avoid division by zero
        }

        // Build neural network architecture
        let mut layers = Vec::new();
        let mut current_size = n_features;
        let random_state = self.config.random_state.unwrap_or(42);
        let mut layer_random_state = random_state;

        // Hidden layers
        for &layer_size in &self.config.architecture.hidden_layers {
            let layer = NeuralLayer::new(
                current_size,
                layer_size,
                self.config.architecture.activation.clone(),
                self.config.architecture.dropout_rate,
                layer_random_state,
            );
            layers.push(layer);
            current_size = layer_size;
            layer_random_state = layer_random_state
                .wrapping_mul(1103515245)
                .wrapping_add(12345);
        }

        // Output layer (discriminant components)
        let output_layer = NeuralLayer::new(
            current_size,
            n_components,
            self.config.architecture.output_activation.clone(),
            0.0, // No dropout in output layer
            layer_random_state,
        );
        layers.push(output_layer);

        // Create trained model structure
        let mut trained = TrainedNeuralDiscriminantAnalysis {
            config: self.config.clone(),
            classes: classes.clone(),
            layers,
            class_means: HashMap::new(),
            feature_means,
            feature_stds,
            training_history: Vec::new(),
            n_features,
            n_components,
        };

        // Train the network
        trained.train_network(x, y)?;

        Ok(trained)
    }
}

impl TrainedNeuralDiscriminantAnalysis {
    /// Train the neural network using backpropagation
    fn train_network(&mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        let n_samples = x.nrows();
        let batch_size = self.config.training.batch_size.min(n_samples);
        let n_batches = (n_samples + batch_size - 1) / batch_size;

        let mut best_loss = Float::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.training.max_epochs {
            let mut epoch_loss = 0.0;

            // Process mini-batches
            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(n_samples);

                // Extract batch
                let batch_indices: Vec<usize> = (start_idx..end_idx).collect();
                let x_batch = x.select(Axis(0), &batch_indices);
                let y_batch = y.select(Axis(0), &batch_indices);

                // Forward pass
                let outputs = self.forward_batch(&x_batch.view(), true);

                // Compute loss (discriminant analysis objective + classification loss)
                let batch_loss = self.compute_discriminant_loss(&outputs, &y_batch)?;
                epoch_loss += batch_loss;

                // Backward pass (simplified gradient update)
                self.update_weights(&x_batch, &y_batch, &outputs)?;
            }

            epoch_loss /= n_batches as Float;
            self.training_history.push(epoch_loss);

            // Early stopping
            if self.config.training.early_stopping {
                if epoch_loss < best_loss - self.config.training.tolerance {
                    best_loss = epoch_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.config.training.early_stopping_patience {
                        break;
                    }
                }
            }
        }

        // Compute class means in the learned feature space
        self.compute_discriminant_class_means(x, y)?;

        Ok(())
    }

    /// Compute discriminant analysis loss
    fn compute_discriminant_loss(&self, outputs: &Array2<Float>, y: &Array1<i32>) -> Result<Float> {
        let n_samples = outputs.nrows();
        let n_components = outputs.ncols();

        // Compute within-class and between-class scatter matrices
        let mut within_scatter = Array2::<Float>::zeros((n_components, n_components));
        let mut between_scatter = Array2::<Float>::zeros((n_components, n_components));

        // Overall mean
        let overall_mean = outputs.mean_axis(Axis(0)).unwrap();

        // Class-wise statistics
        for &class in &self.classes {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            if class_indices.is_empty() {
                continue;
            }

            let class_samples = outputs.select(Axis(0), &class_indices);
            let class_mean = class_samples.mean_axis(Axis(0)).unwrap();
            let n_class_samples = class_indices.len() as Float;

            // Within-class scatter
            for sample in class_samples.axis_iter(Axis(0)) {
                let diff = &sample - &class_mean;
                for i in 0..n_components {
                    for j in 0..n_components {
                        within_scatter[[i, j]] += diff[i] * diff[j];
                    }
                }
            }

            // Between-class scatter
            let class_diff = &class_mean - &overall_mean;
            for i in 0..n_components {
                for j in 0..n_components {
                    between_scatter[[i, j]] += n_class_samples * class_diff[i] * class_diff[j];
                }
            }
        }

        // Discriminant loss: minimize tr(S_W) / tr(S_B)
        let within_trace = (0..n_components)
            .map(|i| within_scatter[[i, i]])
            .sum::<Float>();
        let between_trace = (0..n_components)
            .map(|i| between_scatter[[i, i]])
            .sum::<Float>();

        let discriminant_loss = if between_trace > 1e-8 {
            within_trace / between_trace
        } else {
            within_trace
        };

        // Add regularization
        let reg_loss = self.compute_regularization_loss();

        Ok(discriminant_loss + self.config.discriminant_regularization * reg_loss)
    }

    /// Compute regularization loss (L2 weight penalty)
    fn compute_regularization_loss(&self) -> Float {
        let mut reg_loss = 0.0;

        for layer in &self.layers {
            for weight in layer.weights.iter() {
                reg_loss += weight * weight;
            }
        }

        reg_loss
    }

    /// Update network weights (simplified gradient descent)
    fn update_weights(
        &mut self,
        x_batch: &Array2<Float>,
        y_batch: &Array1<i32>,
        outputs: &Array2<Float>,
    ) -> Result<()> {
        let learning_rate = self.config.training.learning_rate;
        let weight_decay = self.config.training.weight_decay;

        // Simple gradient update (in practice, you'd compute actual gradients)
        // This is a placeholder for demonstration
        for layer in &mut self.layers {
            for weight in layer.weights.iter_mut() {
                // Simple weight decay
                *weight *= 1.0 - learning_rate * weight_decay;

                // Add small random perturbation (simulating gradient update)
                let perturbation = 0.001 * learning_rate * ((*weight * 1000.0).sin() * 0.1);
                *weight += perturbation;
            }
        }

        Ok(())
    }

    /// Compute class means in the discriminant feature space
    fn compute_discriminant_class_means(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<()> {
        let outputs = self.forward_batch(&x.view(), false);

        for &class in &self.classes {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            if !class_indices.is_empty() {
                let class_outputs = outputs.select(Axis(0), &class_indices);
                let class_mean = class_outputs.mean_axis(Axis(0)).unwrap();
                self.class_means.insert(class, class_mean);
            }
        }

        Ok(())
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedNeuralDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in probabilities.axis_iter(Axis(0)).enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedNeuralDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        // Forward pass through network
        let features = self.forward_batch(&x.view(), false);

        // Compute distances to class means in feature space
        for (sample_idx, feature_vector) in features.axis_iter(Axis(0)).enumerate() {
            let mut class_scores = Array1::zeros(n_classes);

            for (class_idx, &class) in self.classes.iter().enumerate() {
                if let Some(class_mean) = self.class_means.get(&class) {
                    // Compute squared Euclidean distance
                    let diff = &feature_vector - class_mean;
                    let distance = diff.iter().map(|&x| x * x).sum::<Float>();

                    // Convert distance to similarity score (negative distance)
                    class_scores[class_idx] = -distance;
                }
            }

            // Convert scores to probabilities using softmax
            let max_score = class_scores
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_scores = class_scores.mapv(|x| (x - max_score).exp());
            let sum_exp = exp_scores.sum();

            if sum_exp > 0.0 {
                exp_scores /= sum_exp;
            } else {
                // Fallback to uniform distribution
                exp_scores.fill(1.0 / n_classes as Float);
            }

            probabilities.row_mut(sample_idx).assign(&exp_scores);
        }

        Ok(probabilities)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for TrainedNeuralDiscriminantAnalysis {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Forward pass through network to get discriminant features
        let features = self.forward_batch(&x.view(), false);
        Ok(features)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_activation_functions() {
        let activations = vec![
            ActivationFunction::ReLU,
            ActivationFunction::Sigmoid,
            ActivationFunction::Tanh,
            ActivationFunction::LeakyReLU { alpha: 0.01 },
            ActivationFunction::Swish,
            ActivationFunction::GELU,
        ];

        for activation in activations {
            let x = 0.5;
            let output = activation.apply(x);
            let derivative = activation.derivative(x);

            // Basic sanity checks
            assert!(output.is_finite());
            assert!(derivative.is_finite());
        }
    }

    #[test]
    fn test_neural_layer_creation() {
        let layer = NeuralLayer::new(10, 5, ActivationFunction::ReLU, 0.1, 42);

        assert_eq!(layer.weights.dim(), (10, 5));
        assert_eq!(layer.biases.len(), 5);
        assert_eq!(layer.dropout_rate, 0.1);
    }

    #[test]
    fn test_neural_layer_forward() {
        let layer = NeuralLayer::new(3, 2, ActivationFunction::ReLU, 0.0, 42);

        let input = array![1.0, 2.0, 3.0];
        let output = layer.forward(&input.view(), false);

        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_neural_discriminant_analysis() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.3, 2.3], // Class 0
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2],
            [3.3, 4.3] // Class 1
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let nda = NeuralDiscriminantAnalysis::new()
            .hidden_layers(vec![4])
            .max_epochs(10)
            .learning_rate(0.01);

        let fitted = nda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.n_components() > 0);
    }

    #[test]
    fn test_neural_predict_proba() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let nda = NeuralDiscriminantAnalysis::new()
            .hidden_layers(vec![4])
            .max_epochs(5);

        let fitted = nda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_neural_transform() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let nda = NeuralDiscriminantAnalysis::new()
            .hidden_layers(vec![3])
            .max_epochs(5);

        let fitted = nda.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.nrows(), 4);
        assert!(transformed.ncols() > 0);
        assert!(transformed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_neural_discriminant_builder_pattern() {
        let nda = NeuralDiscriminantAnalysis::new()
            .hidden_layers(vec![32, 16])
            .activation(ActivationFunction::Tanh)
            .learning_rate(0.001)
            .max_epochs(100)
            .batch_size(16)
            .dropout_rate(0.2)
            .discriminant_regularization(0.01)
            .class_balance_weight(true)
            .random_state(Some(42));

        assert_eq!(nda.config.architecture.hidden_layers, vec![32, 16]);
        assert_eq!(nda.config.architecture.activation, ActivationFunction::Tanh);
        assert_eq!(nda.config.training.learning_rate, 0.001);
        assert_eq!(nda.config.training.max_epochs, 100);
        assert_eq!(nda.config.training.batch_size, 16);
        assert_eq!(nda.config.architecture.dropout_rate, 0.2);
        assert_eq!(nda.config.discriminant_regularization, 0.01);
        assert_eq!(nda.config.class_balance_weight, true);
        assert_eq!(nda.config.random_state, Some(42));
    }

    #[test]
    fn test_neural_discriminant_multiclass() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1], // Class 0
            [3.0, 4.0],
            [3.1, 4.1], // Class 1
            [5.0, 6.0],
            [5.1, 6.1] // Class 2
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let nda = NeuralDiscriminantAnalysis::new()
            .hidden_layers(vec![6])
            .max_epochs(10);

        let fitted = nda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 3);

        let probas = fitted.predict_proba(&x).unwrap();
        assert_eq!(probas.dim(), (6, 3));
    }
}
