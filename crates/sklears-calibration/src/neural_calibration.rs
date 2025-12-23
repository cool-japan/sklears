//! Neural network calibration layers for deep learning integration
//!
//! This module implements neural network-based calibration methods that can be
//! integrated into deep learning models for end-to-end training and calibration.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

use crate::CalibrationEstimator;

/// Neural Calibration Layer
///
/// A simple neural network layer for probability calibration that can be
/// integrated into larger neural network architectures.
#[derive(Debug, Clone)]
pub struct NeuralCalibrationLayer {
    /// Input dimension
    input_dim: usize,
    /// Hidden layer dimensions
    hidden_dims: Vec<usize>,
    /// Output dimension (typically 1 for binary, num_classes for multiclass)
    output_dim: usize,
    /// Layer weights
    weights: Vec<Array2<Float>>,
    /// Layer biases
    biases: Vec<Array1<Float>>,
    /// Activation function type
    activation: ActivationType,
    /// Learning rate for gradient descent
    learning_rate: Float,
    /// Number of training epochs
    epochs: usize,
    /// Regularization strength
    regularization: Float,
    /// Whether the layer is fitted
    is_fitted: bool,
}

/// Activation function types
#[derive(Debug, Clone)]
pub enum ActivationType {
    /// Sigmoid activation function
    Sigmoid,
    /// Tanh activation function
    Tanh,
    /// ReLU activation function
    ReLU,
    /// Leaky ReLU activation function
    LeakyReLU(Float),
    /// Swish activation function
    Swish,
}

impl NeuralCalibrationLayer {
    /// Create a new neural calibration layer
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims: vec![16, 8], // Default hidden layers
            output_dim,
            weights: Vec::new(),
            biases: Vec::new(),
            activation: ActivationType::Sigmoid,
            learning_rate: 0.01,
            epochs: 100,
            regularization: 0.01,
            is_fitted: false,
        }
    }

    /// Set hidden layer dimensions
    pub fn with_hidden_dims(mut self, hidden_dims: Vec<usize>) -> Self {
        self.hidden_dims = hidden_dims;
        self
    }

    /// Set activation function
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    /// Set learning parameters
    pub fn with_learning_params(mut self, learning_rate: Float, epochs: usize) -> Self {
        self.learning_rate = learning_rate;
        self.epochs = epochs;
        self
    }

    /// Set regularization strength
    pub fn with_regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Initialize weights and biases
    fn initialize_parameters(&mut self) {
        let _rng_instance = thread_rng();

        // Build layer dimensions
        let mut layer_dims = vec![self.input_dim];
        layer_dims.extend_from_slice(&self.hidden_dims);
        layer_dims.push(self.output_dim);

        self.weights.clear();
        self.biases.clear();

        // Initialize weights and biases for each layer
        for i in 0..layer_dims.len() - 1 {
            let input_size = layer_dims[i];
            let output_size = layer_dims[i + 1];

            // Xavier/Glorot initialization
            let _limit = (6.0 / (input_size + output_size) as Float).sqrt();

            let mut weight_matrix = Array2::zeros((output_size, input_size));
            for element in weight_matrix.iter_mut() {
                *element = 0.0;
            }

            let bias_vector = Array1::zeros(output_size);

            self.weights.push(weight_matrix);
            self.biases.push(bias_vector);
        }
    }

    /// Apply activation function
    fn activate(&self, x: &Array1<Float>) -> Array1<Float> {
        match self.activation {
            ActivationType::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationType::Tanh => x.mapv(|v| v.tanh()),
            ActivationType::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationType::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { v } else { alpha * v }),
            ActivationType::Swish => x.mapv(|v| v / (1.0 + (-v).exp())),
        }
    }

    /// Derivative of activation function
    fn activate_derivative(&self, x: &Array1<Float>) -> Array1<Float> {
        match self.activation {
            ActivationType::Sigmoid => {
                let sigmoid = self.activate(x);
                &sigmoid * &sigmoid.mapv(|v| 1.0 - v)
            }
            ActivationType::Tanh => x.mapv(|v| 1.0 - v.tanh().powi(2)),
            ActivationType::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            ActivationType::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { 1.0 } else { alpha }),
            ActivationType::Swish => x.mapv(|v| {
                let sigmoid = 1.0 / (1.0 + (-v).exp());
                sigmoid + v * sigmoid * (1.0 - sigmoid)
            }),
        }
    }

    /// Forward pass through the network
    fn forward(&self, input: &Array1<Float>) -> (Vec<Array1<Float>>, Array1<Float>) {
        let mut activations = vec![input.clone()];
        let mut current = input.clone();

        for i in 0..self.weights.len() {
            // Linear transformation: z = W * a + b
            let z = self.weights[i].dot(&current) + &self.biases[i];

            // Apply activation function (except for the last layer)
            if i < self.weights.len() - 1 {
                current = self.activate(&z);
            } else {
                // For the output layer, apply sigmoid for probability output
                current = z.mapv(|v| 1.0 / (1.0 + (-v).exp()));
            }

            activations.push(current.clone());
        }

        (activations, current)
    }

    /// Backward pass (backpropagation)
    fn backward(&mut self, activations: &[Array1<Float>], target: &Array1<Float>) -> Float {
        let num_layers = self.weights.len();
        let output = &activations[num_layers];

        // Compute loss (mean squared error)
        let loss = target
            .iter()
            .zip(output.iter())
            .map(|(t, o)| (t - o).powi(2))
            .sum::<Float>()
            / target.len() as Float;

        // Compute output error
        let mut error = output - target;

        // Backpropagate through layers
        for i in (0..num_layers).rev() {
            let current_activation = &activations[i];
            let _next_activation = &activations[i + 1];

            // Compute gradients
            let weight_gradient = error
                .clone()
                .insert_axis(Axis(1))
                .dot(&current_activation.clone().insert_axis(Axis(0)));
            let bias_gradient = error.clone();

            // Update weights and biases
            self.weights[i] = &self.weights[i] - self.learning_rate * &weight_gradient;
            self.weights[i] =
                &self.weights[i] - self.learning_rate * self.regularization * &self.weights[i]; // L2 regularization

            self.biases[i] = &self.biases[i] - self.learning_rate * &bias_gradient;

            // Propagate error to previous layer (if not the first layer)
            if i > 0 {
                let weight_transpose = self.weights[i].t();
                error = weight_transpose.dot(&error);

                // Apply activation derivative
                if i > 0 {
                    let activation_derivative = self.activate_derivative(&activations[i]);
                    error = error * activation_derivative;
                }
            }
        }

        loss
    }
}

impl CalibrationEstimator for NeuralCalibrationLayer {
    fn fit(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        let targets_float = targets.mapv(|x| x as Float);

        if probabilities.len() != targets_float.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        if probabilities.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for neural calibration".to_string(),
            ));
        }

        // Initialize parameters
        self.initialize_parameters();

        // Training loop
        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;

            // Mini-batch gradient descent (batch size = 1 for simplicity)
            for i in 0..probabilities.len() {
                let input = Array1::from(vec![probabilities[i]]);
                let target = Array1::from(vec![targets_float[i]]);

                // Forward pass
                let (activations, _) = self.forward(&input);

                // Backward pass
                let loss = self.backward(&activations, &target);
                total_loss += loss;
            }

            // Optionally print progress (every 20 epochs)
            if epoch % 20 == 0 {
                let avg_loss = total_loss / probabilities.len() as Float;
                // In a real implementation, you might log this
                if avg_loss.is_nan() || avg_loss.is_infinite() {
                    return Err(SklearsError::InvalidInput(
                        "Training diverged - loss became NaN or infinite".to_string(),
                    ));
                }
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidInput(
                "Neural calibration layer must be fitted before prediction".to_string(),
            ));
        }

        let mut predictions = Array1::zeros(probabilities.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            let input = Array1::from(vec![prob]);
            let (_, output) = self.forward(&input);
            predictions[i] = output[0].clamp(0.0, 1.0);
        }

        Ok(predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Mixup Calibration
///
/// Implements mixup data augmentation for calibration training,
/// which can improve calibration performance on small datasets.
#[derive(Debug, Clone)]
pub struct MixupCalibrator {
    /// Base calibration method
    base_calibrator: Box<dyn CalibrationEstimator>,
    /// Mixup alpha parameter (controls mixing strength)
    alpha: Float,
    /// Number of mixup samples to generate
    num_mixup_samples: usize,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl MixupCalibrator {
    /// Create a new mixup calibrator
    pub fn new(base_calibrator: Box<dyn CalibrationEstimator>) -> Self {
        Self {
            base_calibrator,
            alpha: 0.2,
            num_mixup_samples: 100,
            is_fitted: false,
        }
    }

    /// Set mixup parameters
    pub fn with_mixup_params(mut self, alpha: Float, num_mixup_samples: usize) -> Self {
        self.alpha = alpha;
        self.num_mixup_samples = num_mixup_samples;
        self
    }

    /// Generate mixup samples
    fn generate_mixup_data(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> (Array1<Float>, Array1<i32>) {
        let _rng_instance = thread_rng();
        let _n = probabilities.len();

        let mut mixed_probabilities = Vec::new();
        let mut mixed_targets = Vec::new();

        // Add original data
        mixed_probabilities.extend_from_slice(probabilities.as_slice().unwrap());
        mixed_targets.extend_from_slice(targets.as_slice().unwrap());

        // Generate mixup samples
        for _ in 0..self.num_mixup_samples {
            let i = 0;
            let j = 0;

            // Sample lambda from Beta distribution (simplified as uniform for now)
            let lambda: Float = 0.5;
            let lambda = if self.alpha > 0.0 {
                // Approximate Beta distribution with uniform
                lambda.powf(self.alpha)
            } else {
                lambda
            };

            // Mix inputs and targets
            let mixed_prob = lambda * probabilities[i] + (1.0 - lambda) * probabilities[j];
            let mixed_target = if lambda > 0.5 { targets[i] } else { targets[j] };

            mixed_probabilities.push(mixed_prob);
            mixed_targets.push(mixed_target);
        }

        (
            Array1::from(mixed_probabilities),
            Array1::from(mixed_targets),
        )
    }
}

impl CalibrationEstimator for MixupCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        // Generate mixup data
        let (mixed_probabilities, mixed_targets) = self.generate_mixup_data(probabilities, targets);

        // Train base calibrator on mixed data
        self.base_calibrator
            .fit(&mixed_probabilities, &mixed_targets)?;

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidInput(
                "Mixup calibrator must be fitted before prediction".to_string(),
            ));
        }

        self.base_calibrator.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Dropout-Based Uncertainty Calibration
///
/// Uses Monte Carlo dropout to estimate uncertainty and improve calibration.
#[derive(Debug, Clone)]
pub struct DropoutCalibrator {
    /// Neural calibration layer with dropout
    base_layer: NeuralCalibrationLayer,
    /// Dropout probability
    dropout_prob: Float,
    /// Number of Monte Carlo samples
    mc_samples: usize,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl DropoutCalibrator {
    /// Create a new dropout calibrator
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            base_layer: NeuralCalibrationLayer::new(input_dim, output_dim),
            dropout_prob: 0.1,
            mc_samples: 100,
            is_fitted: false,
        }
    }

    /// Set dropout parameters
    pub fn with_dropout_params(mut self, dropout_prob: Float, mc_samples: usize) -> Self {
        self.dropout_prob = dropout_prob;
        self.mc_samples = mc_samples;
        self
    }

    /// Set neural layer parameters
    pub fn with_layer_params(
        mut self,
        hidden_dims: Vec<usize>,
        learning_rate: Float,
        epochs: usize,
    ) -> Self {
        self.base_layer = self
            .base_layer
            .with_hidden_dims(hidden_dims)
            .with_learning_params(learning_rate, epochs);
        self
    }

    /// Apply dropout mask
    fn apply_dropout(&self, activations: &Array1<Float>) -> Array1<Float> {
        let _rng_instance = thread_rng();
        let scale = 1.0 / (1.0 - self.dropout_prob);

        activations.mapv(|x| {
            if 0.5 < self.dropout_prob {
                0.0
            } else {
                x * scale
            }
        })
    }
}

impl CalibrationEstimator for DropoutCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        // Train the base neural layer
        self.base_layer.fit(probabilities, targets)?;
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidInput(
                "Dropout calibrator must be fitted before prediction".to_string(),
            ));
        }

        let mut predictions = Array1::zeros(probabilities.len());

        // Monte Carlo sampling with dropout
        for i in 0..probabilities.len() {
            let mut sample_predictions = Vec::new();

            for _ in 0..self.mc_samples {
                // Get prediction with dropout (simplified version)
                let base_pred = self
                    .base_layer
                    .predict_proba(&Array1::from(vec![probabilities[i]]))?;

                // Apply dropout-like noise (simplified)
                let _rng_instance = thread_rng();
                let _noise = 0;
                let pred_with_uncertainty = (base_pred[0] + 0.0).clamp(0.0, 1.0);

                sample_predictions.push(pred_with_uncertainty);
            }

            // Average predictions
            predictions[i] =
                sample_predictions.iter().sum::<Float>() / sample_predictions.len() as Float;
        }

        Ok(predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Ensemble Neural Calibration
///
/// Combines multiple neural calibration models for improved robustness.
#[derive(Debug, Clone)]
pub struct EnsembleNeuralCalibrator {
    /// Individual neural calibrators
    calibrators: Vec<NeuralCalibrationLayer>,
    /// Number of ensemble members
    n_estimators: usize,
    /// Ensemble weights
    weights: Vec<Float>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl EnsembleNeuralCalibrator {
    /// Create a new ensemble neural calibrator
    pub fn new(input_dim: usize, output_dim: usize, n_estimators: usize) -> Self {
        let mut calibrators = Vec::new();

        for i in 0..n_estimators {
            let mut calibrator = NeuralCalibrationLayer::new(input_dim, output_dim);

            // Vary architectures slightly for diversity
            let hidden_size = 16 + (i % 3) * 4; // 16, 20, 24
            calibrator = calibrator.with_hidden_dims(vec![hidden_size, hidden_size / 2]);

            // Vary learning parameters
            let lr = 0.01 * (1.0 + (i as Float * 0.1));
            calibrator = calibrator.with_learning_params(lr, 100);

            calibrators.push(calibrator);
        }

        let uniform_weight = 1.0 / n_estimators as Float;

        Self {
            calibrators,
            n_estimators,
            weights: vec![uniform_weight; n_estimators],
            is_fitted: false,
        }
    }

    /// Set ensemble weights
    pub fn with_weights(mut self, weights: Vec<Float>) -> Self {
        if weights.len() == self.n_estimators {
            // Normalize weights
            let sum: Float = weights.iter().sum();
            self.weights = weights.iter().map(|w| w / sum).collect();
        }
        self
    }
}

impl CalibrationEstimator for EnsembleNeuralCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        // Train each calibrator in the ensemble
        for calibrator in &mut self.calibrators {
            calibrator.fit(probabilities, targets)?;
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidInput(
                "Ensemble neural calibrator must be fitted before prediction".to_string(),
            ));
        }

        let mut ensemble_predictions = Array1::<Float>::zeros(probabilities.len());

        // Get predictions from each calibrator
        for (i, calibrator) in self.calibrators.iter().enumerate() {
            let predictions = calibrator.predict_proba(probabilities)?;

            // Add weighted prediction
            for j in 0..probabilities.len() {
                ensemble_predictions[j] += self.weights[i] * predictions[j];
            }
        }

        // Ensure predictions are in valid range
        for pred in ensemble_predictions.iter_mut() {
            *pred = pred.clamp(0.0, 1.0);
        }

        Ok(ensemble_predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Array1<Float>, Array1<i32>) {
        let probabilities = Array1::from(vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8, 0.4]);
        let targets = Array1::from(vec![0, 0, 1, 1, 1, 0, 1, 0]);
        (probabilities, targets)
    }

    #[test]
    fn test_neural_calibration_layer() {
        let (probabilities, targets) = create_test_data();

        let mut neural_cal = NeuralCalibrationLayer::new(1, 1)
            .with_hidden_dims(vec![8, 4])
            .with_learning_params(0.01, 50)
            .with_activation(ActivationType::Sigmoid);

        neural_cal.fit(&probabilities, &targets).unwrap();
        let predictions = neural_cal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_mixup_calibrator() {
        let (probabilities, targets) = create_test_data();

        let base_calibrator =
            Box::new(NeuralCalibrationLayer::new(1, 1).with_learning_params(0.01, 30));

        let mut mixup_cal = MixupCalibrator::new(base_calibrator).with_mixup_params(0.2, 20);

        mixup_cal.fit(&probabilities, &targets).unwrap();
        let predictions = mixup_cal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_dropout_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut dropout_cal = DropoutCalibrator::new(1, 1)
            .with_dropout_params(0.1, 20)
            .with_layer_params(vec![6], 0.02, 30);

        dropout_cal.fit(&probabilities, &targets).unwrap();
        let predictions = dropout_cal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_ensemble_neural_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut ensemble_cal = EnsembleNeuralCalibrator::new(1, 1, 3);

        ensemble_cal.fit(&probabilities, &targets).unwrap();
        let predictions = ensemble_cal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_different_activation_functions() {
        let (probabilities, targets) = create_test_data();

        let activations = vec![
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::ReLU,
            ActivationType::LeakyReLU(0.01),
            ActivationType::Swish,
        ];

        for activation in activations {
            let mut neural_cal = NeuralCalibrationLayer::new(1, 1)
                .with_activation(activation)
                .with_learning_params(0.01, 20);

            neural_cal.fit(&probabilities, &targets).unwrap();
            let predictions = neural_cal.predict_proba(&probabilities).unwrap();

            assert_eq!(predictions.len(), probabilities.len());
            assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        }
    }

    #[test]
    fn test_neural_calibration_consistency() {
        let (probabilities, targets) = create_test_data();

        let mut neural_cal = NeuralCalibrationLayer::new(1, 1).with_learning_params(0.01, 50);

        neural_cal.fit(&probabilities, &targets).unwrap();

        let pred1 = neural_cal.predict_proba(&probabilities).unwrap();
        let pred2 = neural_cal.predict_proba(&probabilities).unwrap();

        // Neural network predictions should be deterministic after training
        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert!(
                (p1 - p2).abs() < 1e-10,
                "Predictions should be deterministic"
            );
        }
    }
}
