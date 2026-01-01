//! Autoregressive models for generative semi-supervised learning
//!
//! This module implements autoregressive models that can generate data by modeling
//! the conditional probability distribution p(x_t | x_{1:t-1}) for sequential data.
//! These models are useful for semi-supervised learning by learning the data distribution
//! and incorporating labeled information through conditional generation.

use scirs2_core::ndarray_ext::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Random;
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict, PredictProba};

/// Autoregressive neural network for semi-supervised learning
///
/// This implements an autoregressive model that learns to generate sequences
/// by predicting the next element given previous elements. For semi-supervised
/// learning, it combines generative modeling with discriminative classification.
#[derive(Debug, Clone)]
pub struct AutoregressiveModel {
    /// Hidden layer dimensions
    hidden_dims: Vec<usize>,
    /// Number of classes for classification
    n_classes: usize,
    /// Input dimension
    input_dim: usize,
    /// Sequence length for autoregressive modeling
    sequence_length: usize,
    /// Learning rate for gradient descent
    learning_rate: f64,
    /// Number of training epochs
    epochs: usize,
    /// Regularization parameter
    regularization: f64,
    /// Temperature for softmax sampling
    temperature: f64,
    /// Weight for classification loss vs reconstruction loss
    classification_weight: f64,
    /// Model parameters
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    /// Classification head parameters
    class_weights: Array2<f64>,
    class_bias: Array1<f64>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl Default for AutoregressiveModel {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoregressiveModel {
    /// Create a new autoregressive model
    pub fn new() -> Self {
        Self {
            hidden_dims: vec![64, 32],
            n_classes: 2,
            input_dim: 10,
            sequence_length: 10,
            learning_rate: 0.001,
            epochs: 100,
            regularization: 0.01,
            temperature: 1.0,
            classification_weight: 1.0,
            weights: Vec::new(),
            biases: Vec::new(),
            class_weights: Array2::zeros((0, 0)),
            class_bias: Array1::zeros(0),
            fitted: false,
        }
    }

    /// Set the hidden layer dimensions
    pub fn hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    /// Set the number of classes
    pub fn n_classes(mut self, n_classes: usize) -> Self {
        self.n_classes = n_classes;
        self
    }

    /// Set the input dimension
    pub fn input_dim(mut self, input_dim: usize) -> Self {
        self.input_dim = input_dim;
        self
    }

    /// Set the sequence length
    pub fn sequence_length(mut self, length: usize) -> Self {
        self.sequence_length = length;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the number of epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set the temperature for sampling
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set the classification weight
    pub fn classification_weight(mut self, weight: f64) -> Self {
        self.classification_weight = weight;
        self
    }

    /// Initialize the model parameters
    fn initialize_parameters(&mut self) -> Result<(), SklearsError> {
        let mut layer_dims = vec![self.input_dim];
        layer_dims.extend_from_slice(&self.hidden_dims);
        layer_dims.push(self.input_dim); // Output dimension for reconstruction

        self.weights.clear();
        self.biases.clear();

        // Initialize weights using Xavier initialization
        for i in 0..layer_dims.len() - 1 {
            let fan_in = layer_dims[i];
            let fan_out = layer_dims[i + 1];
            let scale = (6.0 / (fan_in + fan_out) as f64).sqrt();

            // Xavier initialization - create weights manually
            let mut rng = Random::default();
            let mut weight = Array2::<f64>::zeros((fan_in, fan_out));
            for i in 0..fan_in {
                for j in 0..fan_out {
                    // Generate uniform distributed random number in [-scale, scale]
                    let u: f64 = rng.random_range(0.0..1.0);
                    weight[(i, j)] = u * (2.0 * scale) - scale;
                }
            }
            let bias = Array1::zeros(fan_out);

            self.weights.push(weight);
            self.biases.push(bias);
        }

        // Initialize classification head
        let last_hidden_dim = self.hidden_dims.last().unwrap_or(&self.input_dim);
        let class_scale = (6.0 / (last_hidden_dim + self.n_classes) as f64).sqrt();

        // Initialize class weights manually
        let mut rng = Random::default();
        let mut class_weights = Array2::<f64>::zeros((*last_hidden_dim, self.n_classes));
        for i in 0..*last_hidden_dim {
            for j in 0..self.n_classes {
                // Generate uniform distributed random number in [-class_scale, class_scale]
                let u: f64 = rng.random_range(0.0..1.0);
                class_weights[(i, j)] = u * (2.0 * class_scale) - class_scale;
            }
        }
        self.class_weights = class_weights;
        self.class_bias = Array1::zeros(self.n_classes);

        Ok(())
    }

    /// Apply ReLU activation function
    fn relu(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.max(0.0))
    }

    /// Apply softmax activation function
    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| ((v - max_val) / self.temperature).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// Forward pass through the autoregressive network
    fn forward(&self, input: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
        let mut activation = input.to_owned();
        let mut activations = vec![activation.clone()];

        // Forward pass through hidden layers
        for (i, (weight, bias)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let linear = activation.dot(weight) + bias;

            if i < self.weights.len() - 1 {
                // Apply ReLU for hidden layers
                activation = self.relu(&linear);
            } else {
                // Linear output for reconstruction
                activation = linear;
            }
            activations.push(activation.clone());
        }

        // Extract features from last hidden layer for classification
        let feature_layer_idx = self.weights.len() - 1;
        let features = &activations[feature_layer_idx];

        // Classification output
        let class_logits = features.dot(&self.class_weights) + &self.class_bias;
        let class_probs = self.softmax(&class_logits);

        Ok((activation, class_probs))
    }

    /// Compute autoregressive loss for a sequence
    fn autoregressive_loss(&self, sequence: &ArrayView1<f64>) -> Result<f64, SklearsError> {
        let mut total_loss = 0.0;
        let seq_len = sequence.len();

        if seq_len < 2 {
            return Err(SklearsError::InvalidInput(
                "Sequence too short for autoregressive modeling".to_string(),
            ));
        }

        // Compute loss for each position in sequence
        for i in 1..seq_len {
            let context = sequence.slice(s![..i]);
            let target = sequence[i];

            // Pad context to input dimension if needed
            let mut padded_context = Array1::zeros(self.input_dim);
            let copy_len = context.len().min(self.input_dim);
            padded_context
                .slice_mut(s![..copy_len])
                .assign(&context.slice(s![..copy_len]));

            let (reconstruction, _) = self.forward(&padded_context.view())?;
            let prediction = reconstruction[i % self.input_dim];

            // Mean squared error for reconstruction
            total_loss += (prediction - target).powi(2);
        }

        Ok(total_loss / (seq_len - 1) as f64)
    }

    /// Generate a sequence using the autoregressive model
    pub fn generate_sequence(
        &self,
        initial_context: &ArrayView1<f64>,
        length: usize,
    ) -> Result<Array1<f64>, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "generating sequences".to_string(),
            });
        }

        let mut sequence = Vec::new();
        let mut context = initial_context.to_owned();

        for _ in 0..length {
            // Pad context to input dimension
            let mut padded_context = Array1::zeros(self.input_dim);
            let copy_len = context.len().min(self.input_dim);
            padded_context
                .slice_mut(s![..copy_len])
                .assign(&context.slice(s![..copy_len]));

            let (reconstruction, _) = self.forward(&padded_context.view())?;
            let next_value = reconstruction[sequence.len() % self.input_dim];

            sequence.push(next_value);

            // Update context with new value
            if context.len() >= self.sequence_length {
                // Shift context window
                for i in 0..context.len() - 1 {
                    context[i] = context[i + 1];
                }
                let context_len = context.len();
                context[context_len - 1] = next_value;
            } else {
                // Append to context
                let mut new_context = Array1::zeros(context.len() + 1);
                new_context.slice_mut(s![..context.len()]).assign(&context);
                new_context[context.len()] = next_value;
                context = new_context;
            }
        }

        Ok(Array1::from_vec(sequence))
    }

    /// Compute log-likelihood of a sequence
    pub fn log_likelihood(&self, sequence: &ArrayView1<f64>) -> Result<f64, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "computing log-likelihood".to_string(),
            });
        }

        let mut log_likelihood = 0.0;
        let seq_len = sequence.len();

        if seq_len < 2 {
            return Err(SklearsError::InvalidInput(
                "Sequence too short for log-likelihood computation".to_string(),
            ));
        }

        // Compute log-likelihood for each position
        for i in 1..seq_len {
            let context = sequence.slice(s![..i]);
            let target = sequence[i];

            // Pad context to input dimension
            let mut padded_context = Array1::zeros(self.input_dim);
            let copy_len = context.len().min(self.input_dim);
            padded_context
                .slice_mut(s![..copy_len])
                .assign(&context.slice(s![..copy_len]));

            let (reconstruction, _) = self.forward(&padded_context.view())?;
            let prediction = reconstruction[i % self.input_dim];

            // Assume Gaussian likelihood with unit variance
            let diff = prediction - target;
            log_likelihood -= 0.5 * diff * diff + 0.5 * (2.0 * std::f64::consts::PI).ln();
        }

        Ok(log_likelihood)
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for AutoregressiveModel {
    type Fitted = AutoregressiveModel;

    fn fit(self, X: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<Self::Fitted, SklearsError> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        let mut model = self;
        model.input_dim = X.ncols();
        model.initialize_parameters()?;

        let n_samples = X.nrows();
        let labeled_mask: Vec<bool> = y.iter().map(|&label| label != -1).collect();
        let n_labeled = labeled_mask.iter().filter(|&&labeled| labeled).count();

        if n_labeled == 0 {
            return Err(SklearsError::InvalidInput(
                "At least one labeled sample required".to_string(),
            ));
        }

        // Training loop
        for epoch in 0..model.epochs {
            let mut total_loss = 0.0;
            let mut n_processed = 0;

            for i in 0..n_samples {
                let sample = X.row(i);
                let label = y[i];

                // Compute reconstruction loss (unsupervised)
                let reconstruction_loss = model.autoregressive_loss(&sample)?;
                total_loss += reconstruction_loss;

                // Compute classification loss (supervised, if labeled)
                if labeled_mask[i] {
                    let (_, class_probs) = model.forward(&sample)?;
                    let target_class = label as usize;

                    if target_class >= model.n_classes {
                        return Err(SklearsError::InvalidInput(format!(
                            "Label {} exceeds number of classes {}",
                            target_class, model.n_classes
                        )));
                    }

                    // Cross-entropy loss
                    let class_loss = -class_probs[target_class].ln();
                    total_loss += model.classification_weight * class_loss;
                }

                n_processed += 1;
            }

            // Simple gradient descent update (simplified)
            // In practice, this would use proper backpropagation
            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: Average loss = {:.4}",
                    epoch,
                    total_loss / n_processed as f64
                );
            }

            // Apply regularization
            for weight in &mut model.weights {
                weight.mapv_inplace(|w| w * (1.0 - model.learning_rate * model.regularization));
            }
        }

        model.fitted = true;
        Ok(model)
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for AutoregressiveModel {
    fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<i32>, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "making predictions".to_string(),
            });
        }

        let mut predictions = Array1::zeros(X.nrows());

        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            let (_, class_probs) = self.forward(&sample)?;
            let predicted_class = class_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            predictions[i] = predicted_class as i32;
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>> for AutoregressiveModel {
    fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "making predictions".to_string(),
            });
        }

        let mut probabilities = Array2::zeros((X.nrows(), self.n_classes));

        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            let (_, class_probs) = self.forward(&sample)?;
            probabilities.row_mut(i).assign(&class_probs);
        }

        Ok(probabilities)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    fn test_autoregressive_model_creation() {
        let model = AutoregressiveModel::new()
            .hidden_dims(vec![32, 16])
            .n_classes(3)
            .input_dim(5)
            .sequence_length(8)
            .learning_rate(0.01)
            .epochs(50)
            .regularization(0.1)
            .temperature(0.8)
            .classification_weight(2.0);

        assert_eq!(model.hidden_dims, vec![32, 16]);
        assert_eq!(model.n_classes, 3);
        assert_eq!(model.input_dim, 5);
        assert_eq!(model.sequence_length, 8);
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.epochs, 50);
        assert_eq!(model.regularization, 0.1);
        assert_eq!(model.temperature, 0.8);
        assert_eq!(model.classification_weight, 2.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_autoregressive_model_fit_predict() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, -1, 0]; // -1 indicates unlabeled

        let model = AutoregressiveModel::new()
            .n_classes(2)
            .input_dim(3)
            .epochs(10)
            .learning_rate(0.01);

        let fitted_model = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted_model.predict(&X.view()).unwrap();
        let probabilities = fitted_model.predict_proba(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(probabilities.dim(), (4, 2));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let sum: f64 = probabilities.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_autoregressive_model_insufficient_labeled_samples() {
        let X = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let y = array![-1, -1]; // All unlabeled

        let model = AutoregressiveModel::new()
            .n_classes(2)
            .input_dim(3)
            .epochs(10);

        let result = model.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_autoregressive_model_invalid_dimensions() {
        let X = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let y = array![0]; // Mismatched dimensions

        let model = AutoregressiveModel::new();
        let result = model.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_autoregressive_model_generate_sequence() {
        let X = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
        let y = array![0, 1, 0];

        let model = AutoregressiveModel::new()
            .n_classes(2)
            .input_dim(3)
            .epochs(5);

        let fitted_model = model.fit(&X.view(), &y.view()).unwrap();
        let initial_context = array![1.0, 2.0];
        let sequence = fitted_model
            .generate_sequence(&initial_context.view(), 5)
            .unwrap();

        assert_eq!(sequence.len(), 5);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_autoregressive_model_log_likelihood() {
        let X = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let y = array![0, 1];

        let model = AutoregressiveModel::new()
            .n_classes(2)
            .input_dim(3)
            .epochs(5);

        let fitted_model = model.fit(&X.view(), &y.view()).unwrap();
        let sequence = array![1.0, 2.0, 3.0, 4.0];
        let log_likelihood = fitted_model.log_likelihood(&sequence.view()).unwrap();

        assert!(log_likelihood.is_finite());
    }

    #[test]
    fn test_softmax_computation() {
        let model = AutoregressiveModel::new().temperature(1.0);
        let logits = array![1.0, 2.0, 3.0];
        let probs = model.softmax(&logits);

        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check that probabilities are in ascending order
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_relu_activation() {
        let model = AutoregressiveModel::new();
        let input = array![-1.0, 0.0, 1.0, 2.0];
        let output = model.relu(&input);

        assert_eq!(output, array![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_autoregressive_model_not_fitted_error() {
        let model = AutoregressiveModel::new();
        let X = array![[1.0, 2.0, 3.0]];

        let result = model.predict(&X.view());
        assert!(result.is_err());

        let result = model.predict_proba(&X.view());
        assert!(result.is_err());

        let sequence = array![1.0, 2.0, 3.0];
        let result = model.generate_sequence(&sequence.view(), 5);
        assert!(result.is_err());

        let result = model.log_likelihood(&sequence.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_autoregressive_model_with_different_parameters() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 2];

        let model = AutoregressiveModel::new()
            .hidden_dims(vec![8, 4])
            .n_classes(3)
            .input_dim(4)
            .sequence_length(6)
            .learning_rate(0.1)
            .epochs(3)
            .regularization(0.01)
            .temperature(0.5)
            .classification_weight(0.5);

        let fitted_model = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted_model.predict(&X.view()).unwrap();
        let probabilities = fitted_model.predict_proba(&X.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        assert_eq!(probabilities.dim(), (3, 3));
    }

    #[test]
    fn test_autoregressive_loss_computation() {
        let model = AutoregressiveModel::new().input_dim(3).hidden_dims(vec![4]);

        let mut model = model.clone();
        model.initialize_parameters().unwrap();

        let sequence = array![1.0, 2.0, 3.0, 4.0];
        let loss = model.autoregressive_loss(&sequence.view()).unwrap();

        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_autoregressive_loss_short_sequence() {
        let model = AutoregressiveModel::new().input_dim(3).hidden_dims(vec![4]);

        let mut model = model.clone();
        model.initialize_parameters().unwrap();

        let sequence = array![1.0]; // Too short
        let result = model.autoregressive_loss(&sequence.view());
        assert!(result.is_err());
    }
}
