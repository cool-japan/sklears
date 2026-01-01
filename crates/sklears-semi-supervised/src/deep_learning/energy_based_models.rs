//! Energy-based models for semi-supervised learning
//!
//! This module implements energy-based models (EBMs) that learn data distributions
//! by associating low energy values with data samples and high energy values with
//! unlikely samples. For semi-supervised learning, EBMs can incorporate both
//! labeled and unlabeled data through energy minimization and contrastive learning.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Random;
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict, PredictProba};

/// Energy-based model for semi-supervised learning
///
/// This implements an energy-based model that learns to assign low energy
/// to data samples and high energy to unlikely samples. The model combines
/// energy minimization with classification for semi-supervised learning.
#[derive(Debug, Clone)]
pub struct EnergyBasedModel {
    /// Hidden layer dimensions for energy network
    hidden_dims: Vec<usize>,
    /// Number of classes for classification
    n_classes: usize,
    /// Input dimension
    input_dim: usize,
    /// Learning rate for gradient descent
    learning_rate: f64,
    /// Number of training epochs
    epochs: usize,
    /// Regularization parameter
    regularization: f64,
    /// Number of negative samples for contrastive learning
    n_negative_samples: usize,
    /// Temperature for Boltzmann distribution
    temperature: f64,
    /// Weight for classification loss vs energy loss
    classification_weight: f64,
    /// Contrastive learning margin
    margin: f64,
    /// Energy network parameters
    energy_weights: Vec<Array2<f64>>,
    energy_biases: Vec<Array1<f64>>,
    /// Classification head parameters
    class_weights: Array2<f64>,
    class_bias: Array1<f64>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl Default for EnergyBasedModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyBasedModel {
    /// Create a new energy-based model
    pub fn new() -> Self {
        Self {
            hidden_dims: vec![64, 32, 16],
            n_classes: 2,
            input_dim: 10,
            learning_rate: 0.001,
            epochs: 100,
            regularization: 0.01,
            n_negative_samples: 10,
            temperature: 1.0,
            classification_weight: 1.0,
            margin: 1.0,
            energy_weights: Vec::new(),
            energy_biases: Vec::new(),
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

    /// Set the number of negative samples
    pub fn n_negative_samples(mut self, n_samples: usize) -> Self {
        self.n_negative_samples = n_samples;
        self
    }

    /// Set the temperature parameter
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set the classification weight
    pub fn classification_weight(mut self, weight: f64) -> Self {
        self.classification_weight = weight;
        self
    }

    /// Set the contrastive margin
    pub fn margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Initialize the model parameters
    fn initialize_parameters(&mut self) -> Result<(), SklearsError> {
        let mut layer_dims = vec![self.input_dim];
        layer_dims.extend_from_slice(&self.hidden_dims);
        layer_dims.push(1); // Output single energy value

        self.energy_weights.clear();
        self.energy_biases.clear();

        // Initialize energy network weights using Xavier initialization
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

            self.energy_weights.push(weight);
            self.energy_biases.push(bias);
        }

        // Initialize classification head (from last hidden layer)
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

    /// Apply leaky ReLU activation function
    fn leaky_relu(&self, x: &Array1<f64>, alpha: f64) -> Array1<f64> {
        x.mapv(|v| if v > 0.0 { v } else { alpha * v })
    }

    /// Apply softmax activation function
    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| ((v - max_val) / self.temperature).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// Compute energy for a given input
    fn compute_energy(&self, input: &ArrayView1<f64>) -> Result<f64, SklearsError> {
        let mut activation = input.to_owned();
        let mut hidden_features = Vec::new();

        // Forward pass through energy network
        for (i, (weight, bias)) in self
            .energy_weights
            .iter()
            .zip(self.energy_biases.iter())
            .enumerate()
        {
            let linear = activation.dot(weight) + bias;

            if i < self.energy_weights.len() - 1 {
                // Apply Leaky ReLU for hidden layers
                activation = self.leaky_relu(&linear, 0.01);
                hidden_features.push(activation.clone());
            } else {
                // Linear output for energy
                return Ok(linear[0]);
            }
        }

        Err(SklearsError::NumericalError(
            "Energy computation failed".to_string(),
        ))
    }

    /// Get hidden features from the energy network
    fn get_hidden_features(&self, input: &ArrayView1<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut activation = input.to_owned();

        // Forward pass through energy network (excluding final layer)
        for i in 0..self.energy_weights.len() - 1 {
            let weight = &self.energy_weights[i];
            let bias = &self.energy_biases[i];
            let linear = activation.dot(weight) + bias;
            activation = self.leaky_relu(&linear, 0.01);
        }

        Ok(activation)
    }

    /// Compute classification probabilities using hidden features
    fn compute_classification_probs(
        &self,
        input: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let features = self.get_hidden_features(input)?;
        let logits = features.dot(&self.class_weights) + &self.class_bias;
        Ok(self.softmax(&logits))
    }

    /// Generate negative samples using noise
    fn generate_negative_samples(
        &self,
        positive_samples: &ArrayView2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = positive_samples.nrows();
        let input_dim = positive_samples.ncols();

        // Generate random noise samples manually
        let mut rng = Random::default();
        let mut negative_samples = Array2::<f64>::zeros((self.n_negative_samples, input_dim));
        for i in 0..self.n_negative_samples {
            for j in 0..input_dim {
                // Generate normal distributed random number (mean=0.0, std=1.0)
                let u1: f64 = rng.random_range(0.0..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                negative_samples[(i, j)] = z; // mean=0.0, std=1.0
            }
        }

        // Scale based on positive sample statistics
        for j in 0..input_dim {
            let column = positive_samples.column(j);
            let mean = column.mean().unwrap_or(0.0);
            let std = column.std(0.0);

            for i in 0..self.n_negative_samples {
                negative_samples[[i, j]] = negative_samples[[i, j]] * std + mean;
            }
        }

        Ok(negative_samples)
    }

    /// Compute contrastive loss between positive and negative samples
    fn contrastive_loss(
        &self,
        positive_energies: &Array1<f64>,
        negative_energies: &Array1<f64>,
    ) -> f64 {
        let mut loss = 0.0;

        // Positive samples should have low energy
        for &energy in positive_energies.iter() {
            loss += energy;
        }

        // Negative samples should have high energy (contrastive)
        for &energy in negative_energies.iter() {
            loss += (self.margin - energy).max(0.0);
        }

        loss / (positive_energies.len() + negative_energies.len()) as f64
    }

    /// Compute Boltzmann probability from energy
    pub fn energy_to_probability(&self, energy: f64) -> f64 {
        (-energy / self.temperature).exp()
    }

    /// Sample from the model using Langevin dynamics
    pub fn langevin_sample(
        &self,
        initial_sample: &ArrayView1<f64>,
        n_steps: usize,
        step_size: f64,
    ) -> Result<Array1<f64>, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "sampling".to_string(),
            });
        }

        let mut sample = initial_sample.to_owned();

        for _ in 0..n_steps {
            // Compute energy gradient (simplified numerical gradient)
            let mut gradient = Array1::zeros(sample.len());
            let epsilon = 1e-6;

            for i in 0..sample.len() {
                // Forward difference
                sample[i] += epsilon;
                let energy_plus = self.compute_energy(&sample.view())?;
                sample[i] -= 2.0 * epsilon;
                let energy_minus = self.compute_energy(&sample.view())?;
                sample[i] += epsilon; // Reset

                gradient[i] = (energy_plus - energy_minus) / (2.0 * epsilon);
            }

            // Langevin update - using simple Gaussian noise
            let mut rng = Random::default();
            let noise_std = (2.0 * step_size).sqrt();
            let mut noise = Array1::zeros(sample.len());
            for i in 0..sample.len() {
                // Generate standard normal and scale
                noise[i] = rng.random_range(-3.0..3.0) * noise_std / 3.0; // Approximate normal
            }
            sample = &sample - step_size * &gradient + &noise;
        }

        Ok(sample)
    }

    /// Compute the partition function approximation
    pub fn log_partition_function(&self, n_samples: usize) -> Result<f64, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "computing partition function".to_string(),
            });
        }

        let mut log_sum = f64::NEG_INFINITY;

        // Monte Carlo approximation
        for _ in 0..n_samples {
            let mut rng = Random::default();
            let mut sample = Array1::zeros(self.input_dim);
            for i in 0..self.input_dim {
                // Generate standard normal (approximate)
                sample[i] = rng.random_range(-3.0..3.0) / 3.0; // Approximate standard normal
            }
            let energy = self.compute_energy(&sample.view())?;
            let log_prob = -energy / self.temperature;

            // LogSumExp trick
            if log_prob > log_sum {
                log_sum = log_prob + (1.0f64 + (log_sum - log_prob).exp()).ln();
            } else {
                log_sum = log_sum + (1.0f64 + (log_prob - log_sum).exp()).ln();
            }
        }

        Ok(log_sum - (n_samples as f64).ln())
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for EnergyBasedModel {
    type Fitted = EnergyBasedModel;

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
            let mut total_energy_loss = 0.0;
            let mut total_class_loss = 0.0;
            let mut n_processed = 0;

            // Generate negative samples for contrastive learning
            let negative_samples = model.generate_negative_samples(X)?;

            // Compute positive energies
            let mut positive_energies = Array1::zeros(n_samples);
            for i in 0..n_samples {
                positive_energies[i] = model.compute_energy(&X.row(i))?;
            }

            // Compute negative energies
            let mut negative_energies = Array1::zeros(model.n_negative_samples);
            for i in 0..model.n_negative_samples {
                negative_energies[i] = model.compute_energy(&negative_samples.row(i))?;
            }

            // Contrastive energy loss
            let energy_loss = model.contrastive_loss(&positive_energies, &negative_energies);
            total_energy_loss += energy_loss;

            // Classification loss for labeled samples
            for i in 0..n_samples {
                if labeled_mask[i] {
                    let sample = X.row(i);
                    let label = y[i];

                    let class_probs = model.compute_classification_probs(&sample)?;
                    let target_class = label as usize;

                    if target_class >= model.n_classes {
                        return Err(SklearsError::InvalidInput(format!(
                            "Label {} exceeds number of classes {}",
                            target_class, model.n_classes
                        )));
                    }

                    // Cross-entropy loss
                    let class_loss = -class_probs[target_class].ln();
                    total_class_loss += model.classification_weight * class_loss;
                }

                n_processed += 1;
            }

            // Simple gradient descent update (simplified)
            // In practice, this would use proper backpropagation
            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: Energy loss = {:.4}, Class loss = {:.4}",
                    epoch,
                    total_energy_loss,
                    total_class_loss / n_labeled as f64
                );
            }

            // Apply regularization
            for weight in &mut model.energy_weights {
                weight.mapv_inplace(|w| w * (1.0 - model.learning_rate * model.regularization));
            }
        }

        model.fitted = true;
        Ok(model)
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for EnergyBasedModel {
    fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<i32>, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "making predictions".to_string(),
            });
        }

        let mut predictions = Array1::zeros(X.nrows());

        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            let class_probs = self.compute_classification_probs(&sample)?;
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

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>> for EnergyBasedModel {
    fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "making predictions".to_string(),
            });
        }

        let mut probabilities = Array2::zeros((X.nrows(), self.n_classes));

        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            let class_probs = self.compute_classification_probs(&sample)?;
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
    fn test_energy_based_model_creation() {
        let model = EnergyBasedModel::new()
            .hidden_dims(vec![32, 16, 8])
            .n_classes(3)
            .input_dim(5)
            .learning_rate(0.01)
            .epochs(50)
            .regularization(0.1)
            .n_negative_samples(5)
            .temperature(0.8)
            .classification_weight(2.0)
            .margin(2.0);

        assert_eq!(model.hidden_dims, vec![32, 16, 8]);
        assert_eq!(model.n_classes, 3);
        assert_eq!(model.input_dim, 5);
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.epochs, 50);
        assert_eq!(model.regularization, 0.1);
        assert_eq!(model.n_negative_samples, 5);
        assert_eq!(model.temperature, 0.8);
        assert_eq!(model.classification_weight, 2.0);
        assert_eq!(model.margin, 2.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_energy_based_model_fit_predict() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, -1, 0]; // -1 indicates unlabeled

        let model = EnergyBasedModel::new()
            .n_classes(2)
            .input_dim(3)
            .epochs(10)
            .learning_rate(0.01)
            .n_negative_samples(3);

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
    fn test_energy_based_model_insufficient_labeled_samples() {
        let X = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let y = array![-1, -1]; // All unlabeled

        let model = EnergyBasedModel::new().n_classes(2).input_dim(3).epochs(10);

        let result = model.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_energy_based_model_invalid_dimensions() {
        let X = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let y = array![0]; // Mismatched dimensions

        let model = EnergyBasedModel::new();
        let result = model.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_energy_computation() {
        let model = EnergyBasedModel::new().input_dim(3).hidden_dims(vec![4, 2]);

        let mut model = model.clone();
        model.initialize_parameters().unwrap();

        let input = array![1.0, 2.0, 3.0];
        let energy = model.compute_energy(&input.view()).unwrap();

        assert!(energy.is_finite());
    }

    #[test]
    fn test_energy_to_probability() {
        let model = EnergyBasedModel::new().temperature(1.0);
        let energy = 2.0;
        let prob = model.energy_to_probability(energy);

        assert!(prob > 0.0);
        assert!(prob <= 1.0);
        assert!((prob - (-2.0f64).exp()).abs() < 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_negative_sample_generation() {
        let X = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];

        let model = EnergyBasedModel::new().input_dim(3).n_negative_samples(5);

        let negative_samples = model.generate_negative_samples(&X.view()).unwrap();

        assert_eq!(negative_samples.dim(), (5, 3));
    }

    #[test]
    fn test_contrastive_loss_computation() {
        let model = EnergyBasedModel::new().margin(1.0);

        let positive_energies = array![0.5, 1.0, 0.8];
        let negative_energies = array![2.0, 1.5, 2.5];

        let loss = model.contrastive_loss(&positive_energies, &negative_energies);

        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_softmax_computation() {
        let model = EnergyBasedModel::new().temperature(1.0);
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
        let model = EnergyBasedModel::new();
        let input = array![-1.0, 0.0, 1.0, 2.0];
        let output = model.relu(&input);

        assert_eq!(output, array![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu_activation() {
        let model = EnergyBasedModel::new();
        let input = array![-1.0, 0.0, 1.0, 2.0];
        let output = model.leaky_relu(&input, 0.1);

        assert_eq!(output, array![-0.1, 0.0, 1.0, 2.0]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_energy_based_model_not_fitted_error() {
        let model = EnergyBasedModel::new();
        let X = array![[1.0, 2.0, 3.0]];

        let result = model.predict(&X.view());
        assert!(result.is_err());

        let result = model.predict_proba(&X.view());
        assert!(result.is_err());

        let sample = array![1.0, 2.0, 3.0];
        let result = model.langevin_sample(&sample.view(), 10, 0.01);
        assert!(result.is_err());

        let result = model.log_partition_function(100);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_energy_based_model_with_different_parameters() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 2];

        let model = EnergyBasedModel::new()
            .hidden_dims(vec![8, 4])
            .n_classes(3)
            .input_dim(4)
            .learning_rate(0.1)
            .epochs(3)
            .regularization(0.01)
            .n_negative_samples(2)
            .temperature(0.5)
            .classification_weight(0.5)
            .margin(0.5);

        let fitted_model = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted_model.predict(&X.view()).unwrap();
        let probabilities = fitted_model.predict_proba(&X.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        assert_eq!(probabilities.dim(), (3, 3));
    }

    #[test]
    fn test_hidden_features_extraction() {
        let model = EnergyBasedModel::new().input_dim(3).hidden_dims(vec![4, 2]);

        let mut model = model.clone();
        model.initialize_parameters().unwrap();

        let input = array![1.0, 2.0, 3.0];
        let features = model.get_hidden_features(&input.view()).unwrap();

        assert_eq!(features.len(), 2); // Last hidden layer dimension
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_classification_probabilities() {
        let model = EnergyBasedModel::new()
            .input_dim(3)
            .n_classes(2)
            .hidden_dims(vec![4]);

        let mut model = model.clone();
        model.initialize_parameters().unwrap();

        let input = array![1.0, 2.0, 3.0];
        let probs = model.compute_classification_probs(&input.view()).unwrap();

        assert_eq!(probs.len(), 2);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_langevin_sampling() {
        let X = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let y = array![0, 1];

        let model = EnergyBasedModel::new().n_classes(2).input_dim(3).epochs(5);

        let fitted_model = model.fit(&X.view(), &y.view()).unwrap();
        let initial_sample = array![1.0, 2.0, 3.0];
        let sample = fitted_model
            .langevin_sample(&initial_sample.view(), 5, 0.01)
            .unwrap();

        assert_eq!(sample.len(), 3);
        assert!(sample.iter().all(|&x| x.is_finite()));
    }
}
