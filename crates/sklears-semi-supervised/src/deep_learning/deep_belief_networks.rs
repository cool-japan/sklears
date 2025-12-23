//! Deep Belief Networks for semi-supervised learning
//!
//! This module implements Deep Belief Networks (DBNs) which are generative models
//! consisting of multiple layers of Restricted Boltzmann Machines (RBMs).
//! DBNs can be used for semi-supervised learning by pre-training on unlabeled data
//! and fine-tuning with labeled data.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict, PredictProba};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeepBeliefNetworkError {
    #[error("Invalid layer size: {0}")]
    InvalidLayerSize(usize),
    #[error("Invalid learning rate: {0}")]
    InvalidLearningRate(f64),
    #[error("Invalid number of epochs: {0}")]
    InvalidEpochs(usize),
    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(usize),
    #[error("Invalid number of gibbs steps: {0}")]
    InvalidGibbsSteps(usize),
    #[error("Empty hidden layers")]
    EmptyHiddenLayers,
    #[error("Insufficient labeled samples")]
    InsufficientLabeledSamples,
    #[error("Matrix operation failed: {0}")]
    MatrixOperationFailed(String),
    #[error("RBM training failed: {0}")]
    RBMTrainingFailed(String),
}

impl From<DeepBeliefNetworkError> for SklearsError {
    fn from(err: DeepBeliefNetworkError) -> Self {
        SklearsError::FitError(err.to_string())
    }
}

/// Restricted Boltzmann Machine (RBM) component
///
/// An RBM is a two-layer neural network with visible and hidden units
/// that can learn probability distributions over its inputs.
#[derive(Debug, Clone)]
pub struct RestrictedBoltzmannMachine {
    /// n_visible
    pub n_visible: usize,
    /// n_hidden
    pub n_hidden: usize,
    /// learning_rate
    pub learning_rate: f64,
    /// n_epochs
    pub n_epochs: usize,
    /// batch_size
    pub batch_size: usize,
    /// n_gibbs_steps
    pub n_gibbs_steps: usize,
    /// random_state
    pub random_state: Option<u64>,
    weights: Array2<f64>,
    visible_bias: Array1<f64>,
    hidden_bias: Array1<f64>,
}

impl RestrictedBoltzmannMachine {
    pub fn new(n_visible: usize, n_hidden: usize) -> Result<Self> {
        if n_visible == 0 {
            return Err(DeepBeliefNetworkError::InvalidLayerSize(n_visible).into());
        }
        if n_hidden == 0 {
            return Err(DeepBeliefNetworkError::InvalidLayerSize(n_hidden).into());
        }

        Ok(Self {
            n_visible,
            n_hidden,
            learning_rate: 0.01,
            n_epochs: 10,
            batch_size: 32,
            n_gibbs_steps: 1,
            random_state: None,
            weights: Array2::zeros((n_visible, n_hidden)),
            visible_bias: Array1::zeros(n_visible),
            hidden_bias: Array1::zeros(n_hidden),
        })
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(DeepBeliefNetworkError::InvalidLearningRate(learning_rate).into());
        }
        self.learning_rate = learning_rate;
        Ok(self)
    }

    pub fn n_epochs(mut self, n_epochs: usize) -> Result<Self> {
        if n_epochs == 0 {
            return Err(DeepBeliefNetworkError::InvalidEpochs(n_epochs).into());
        }
        self.n_epochs = n_epochs;
        Ok(self)
    }

    pub fn batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(DeepBeliefNetworkError::InvalidBatchSize(batch_size).into());
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    pub fn n_gibbs_steps(mut self, n_gibbs_steps: usize) -> Result<Self> {
        if n_gibbs_steps == 0 {
            return Err(DeepBeliefNetworkError::InvalidGibbsSteps(n_gibbs_steps).into());
        }
        self.n_gibbs_steps = n_gibbs_steps;
        Ok(self)
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    fn initialize_weights(&mut self) -> Result<()> {
        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Initialize weights with small random values manually
        let mut weights = Array2::<f64>::zeros((self.n_visible, self.n_hidden));
        for i in 0..self.n_visible {
            for j in 0..self.n_hidden {
                // Generate normal distributed random number (mean=0.0, std=0.01)
                let u1: f64 = rng.random_range(0.0, 1.0);
                let u2: f64 = rng.random_range(0.0, 1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                weights[(i, j)] = z * 0.01;
            }
        }
        self.weights = weights;
        self.visible_bias = Array1::zeros(self.n_visible);
        self.hidden_bias = Array1::zeros(self.n_hidden);

        Ok(())
    }

    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sample_hidden<R>(
        &self,
        visible: &ArrayView1<f64>,
        rng: &mut Random<R>,
    ) -> Result<Array1<f64>>
    where
        R: scirs2_core::random::Rng,
    {
        let mut hidden_probs = Array1::zeros(self.n_hidden);

        for j in 0..self.n_hidden {
            let mut activation = self.hidden_bias[j];
            for i in 0..self.n_visible {
                activation += visible[i] * self.weights[[i, j]];
            }
            hidden_probs[j] = self.sigmoid(activation);
        }

        // Sample from Bernoulli distribution
        let mut hidden_sample = Array1::zeros(self.n_hidden);
        for j in 0..self.n_hidden {
            let random_val = rng.random_range(0.0, 1.0);
            hidden_sample[j] = if random_val < hidden_probs[j] {
                1.0
            } else {
                0.0
            };
        }

        Ok(hidden_sample)
    }

    fn sample_visible<R>(
        &self,
        hidden: &ArrayView1<f64>,
        rng: &mut Random<R>,
    ) -> Result<Array1<f64>>
    where
        R: scirs2_core::random::Rng,
    {
        let mut visible_probs = Array1::zeros(self.n_visible);

        for i in 0..self.n_visible {
            let mut activation = self.visible_bias[i];
            for j in 0..self.n_hidden {
                activation += hidden[j] * self.weights[[i, j]];
            }
            visible_probs[i] = self.sigmoid(activation);
        }

        // Sample from Bernoulli distribution
        let mut visible_sample = Array1::zeros(self.n_visible);
        for i in 0..self.n_visible {
            let random_val = rng.random_range(0.0, 1.0);
            visible_sample[i] = if random_val < visible_probs[i] {
                1.0
            } else {
                0.0
            };
        }

        Ok(visible_sample)
    }

    fn contrastive_divergence(&mut self, data: &ArrayView2<f64>) -> Result<f64> {
        let n_samples = data.dim().0;
        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let mut total_error = 0.0;

        // Process data in batches
        for batch_start in (0..n_samples).step_by(self.batch_size) {
            let batch_end = std::cmp::min(batch_start + self.batch_size, n_samples);
            let batch_size = batch_end - batch_start;

            if batch_size == 0 {
                continue;
            }

            let mut pos_weights_grad: Array2<f64> = Array2::zeros((self.n_visible, self.n_hidden));
            let mut neg_weights_grad: Array2<f64> = Array2::zeros((self.n_visible, self.n_hidden));
            let mut pos_visible_grad: Array1<f64> = Array1::zeros(self.n_visible);
            let mut neg_visible_grad: Array1<f64> = Array1::zeros(self.n_visible);
            let mut pos_hidden_grad: Array1<f64> = Array1::zeros(self.n_hidden);
            let mut neg_hidden_grad: Array1<f64> = Array1::zeros(self.n_hidden);

            for sample_idx in batch_start..batch_end {
                let visible_data = data.row(sample_idx);

                // Positive phase: clamp visible units to data
                let hidden_probs_pos = self.compute_hidden_probs(&visible_data)?;

                // Negative phase: Gibbs sampling
                let mut visible_sample = visible_data.to_owned();
                let mut hidden_sample = self.sample_hidden(&visible_sample.view(), &mut rng)?;

                for _ in 0..self.n_gibbs_steps {
                    visible_sample = self.sample_visible(&hidden_sample.view(), &mut rng)?;
                    hidden_sample = self.sample_hidden(&visible_sample.view(), &mut rng)?;
                }

                let hidden_probs_neg = self.compute_hidden_probs(&visible_sample.view())?;

                // Accumulate gradients
                for i in 0..self.n_visible {
                    for j in 0..self.n_hidden {
                        pos_weights_grad[[i, j]] += visible_data[i] * hidden_probs_pos[j];
                        neg_weights_grad[[i, j]] += visible_sample[i] * hidden_probs_neg[j];
                    }
                    pos_visible_grad[i] += visible_data[i];
                    neg_visible_grad[i] += visible_sample[i];
                }

                for j in 0..self.n_hidden {
                    pos_hidden_grad[j] += hidden_probs_pos[j];
                    neg_hidden_grad[j] += hidden_probs_neg[j];
                }

                // Compute reconstruction error
                let error: f64 = visible_data
                    .iter()
                    .zip(visible_sample.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                total_error += error;
            }

            // Update parameters
            let lr = self.learning_rate / batch_size as f64;

            self.weights = &self.weights + &((pos_weights_grad - neg_weights_grad) * lr);
            self.visible_bias = &self.visible_bias + &((pos_visible_grad - neg_visible_grad) * lr);
            self.hidden_bias = &self.hidden_bias + &((pos_hidden_grad - neg_hidden_grad) * lr);
        }

        Ok(total_error / n_samples as f64)
    }

    fn compute_hidden_probs(&self, visible: &ArrayView1<f64>) -> Result<Array1<f64>> {
        let mut hidden_probs = Array1::zeros(self.n_hidden);

        for j in 0..self.n_hidden {
            let mut activation = self.hidden_bias[j];
            for i in 0..self.n_visible {
                activation += visible[i] * self.weights[[i, j]];
            }
            hidden_probs[j] = self.sigmoid(activation);
        }

        Ok(hidden_probs)
    }

    pub fn fit(&mut self, data: &ArrayView2<f64>) -> Result<()> {
        self.initialize_weights()?;

        for epoch in 0..self.n_epochs {
            let error = self.contrastive_divergence(data)?;

            if epoch % 10 == 0 {
                println!("RBM Epoch {}: Reconstruction Error = {:.6}", epoch, error);
            }
        }

        Ok(())
    }

    pub fn transform(&self, data: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.dim().0;
        let mut hidden_features = Array2::zeros((n_samples, self.n_hidden));

        for i in 0..n_samples {
            let hidden_probs = self.compute_hidden_probs(&data.row(i))?;
            hidden_features.row_mut(i).assign(&hidden_probs);
        }

        Ok(hidden_features)
    }

    pub fn reconstruct(&self, data: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.dim().0;
        let mut reconstructed = Array2::zeros((n_samples, self.n_visible));
        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        for i in 0..n_samples {
            let hidden_sample = self.sample_hidden(&data.row(i), &mut rng)?;
            let visible_sample = self.sample_visible(&hidden_sample.view(), &mut rng)?;
            reconstructed.row_mut(i).assign(&visible_sample);
        }

        Ok(reconstructed)
    }
}

/// Deep Belief Network for semi-supervised learning
///
/// A DBN consists of multiple RBM layers stacked on top of each other.
/// It uses unsupervised pre-training followed by supervised fine-tuning.
#[derive(Debug, Clone)]
pub struct DeepBeliefNetwork {
    /// hidden_layers
    pub hidden_layers: Vec<usize>,
    /// learning_rate
    pub learning_rate: f64,
    /// pretraining_epochs
    pub pretraining_epochs: usize,
    /// finetuning_epochs
    pub finetuning_epochs: usize,
    /// batch_size
    pub batch_size: usize,
    /// n_gibbs_steps
    pub n_gibbs_steps: usize,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for DeepBeliefNetwork {
    fn default() -> Self {
        Self {
            hidden_layers: vec![100, 50],
            learning_rate: 0.01,
            pretraining_epochs: 50,
            finetuning_epochs: 100,
            batch_size: 32,
            n_gibbs_steps: 1,
            random_state: None,
        }
    }
}

impl DeepBeliefNetwork {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn hidden_layers(mut self, hidden_layers: Vec<usize>) -> Result<Self> {
        if hidden_layers.is_empty() {
            return Err(DeepBeliefNetworkError::EmptyHiddenLayers.into());
        }
        for &size in hidden_layers.iter() {
            if size == 0 {
                return Err(DeepBeliefNetworkError::InvalidLayerSize(size).into());
            }
        }
        self.hidden_layers = hidden_layers;
        Ok(self)
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(DeepBeliefNetworkError::InvalidLearningRate(learning_rate).into());
        }
        self.learning_rate = learning_rate;
        Ok(self)
    }

    pub fn pretraining_epochs(mut self, pretraining_epochs: usize) -> Result<Self> {
        if pretraining_epochs == 0 {
            return Err(DeepBeliefNetworkError::InvalidEpochs(pretraining_epochs).into());
        }
        self.pretraining_epochs = pretraining_epochs;
        Ok(self)
    }

    pub fn finetuning_epochs(mut self, finetuning_epochs: usize) -> Result<Self> {
        if finetuning_epochs == 0 {
            return Err(DeepBeliefNetworkError::InvalidEpochs(finetuning_epochs).into());
        }
        self.finetuning_epochs = finetuning_epochs;
        Ok(self)
    }

    pub fn batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(DeepBeliefNetworkError::InvalidBatchSize(batch_size).into());
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    pub fn n_gibbs_steps(mut self, n_gibbs_steps: usize) -> Result<Self> {
        if n_gibbs_steps == 0 {
            return Err(DeepBeliefNetworkError::InvalidGibbsSteps(n_gibbs_steps).into());
        }
        self.n_gibbs_steps = n_gibbs_steps;
        Ok(self)
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

/// Fitted Deep Belief Network model
#[derive(Debug, Clone)]
pub struct FittedDeepBeliefNetwork {
    /// base_model
    pub base_model: DeepBeliefNetwork,
    /// rbm_layers
    pub rbm_layers: Vec<RestrictedBoltzmannMachine>,
    /// classifier_weights
    pub classifier_weights: Array2<f64>,
    /// classifier_bias
    pub classifier_bias: Array1<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// n_classes
    pub n_classes: usize,
}

impl Estimator for DeepBeliefNetwork {
    type Config = DeepBeliefNetwork;
    type Error = DeepBeliefNetworkError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for DeepBeliefNetwork {
    type Fitted = FittedDeepBeliefNetwork;

    fn fit(self, X: &ArrayView2<'_, f64>, y: &ArrayView1<'_, i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = X.dim();

        // Check for sufficient labeled samples
        let labeled_count = y.iter().filter(|&&label| label != -1).count();
        if labeled_count < 2 {
            return Err(DeepBeliefNetworkError::InsufficientLabeledSamples.into());
        }

        // Get unique classes
        let unique_classes: Vec<i32> = y
            .iter()
            .cloned()
            .filter(|&label| label != -1)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let n_classes = unique_classes.len();

        println!(
            "Starting DBN pre-training with {} layers",
            self.hidden_layers.len()
        );

        // Phase 1: Unsupervised pre-training of RBM layers
        let mut rbm_layers = Vec::new();
        let mut current_input = X.to_owned();

        for (layer_idx, &layer_size) in self.hidden_layers.iter().enumerate() {
            let input_size = current_input.dim().1;

            println!(
                "Pre-training RBM layer {} ({} -> {})",
                layer_idx + 1,
                input_size,
                layer_size
            );

            let mut rbm = RestrictedBoltzmannMachine::new(input_size, layer_size)?
                .learning_rate(self.learning_rate)?
                .n_epochs(self.pretraining_epochs)?
                .batch_size(self.batch_size)?
                .n_gibbs_steps(self.n_gibbs_steps)?;

            if let Some(seed) = self.random_state {
                rbm = rbm.random_state(seed + layer_idx as u64);
            }

            rbm.fit(&current_input.view())?;

            // Transform current input for next layer
            current_input = rbm.transform(&current_input.view())?;

            rbm_layers.push(rbm);
        }

        println!("Pre-training completed. Starting fine-tuning...");

        // Phase 2: Supervised fine-tuning with labeled data
        let labeled_indices: Vec<usize> = y
            .iter()
            .enumerate()
            .filter(|(_, &label)| label != -1)
            .map(|(i, _)| i)
            .collect();

        if labeled_indices.is_empty() {
            return Err(DeepBeliefNetworkError::InsufficientLabeledSamples.into());
        }

        // Extract labeled data
        let labeled_X = Array2::from_shape_vec(
            (labeled_indices.len(), n_features),
            labeled_indices
                .iter()
                .flat_map(|&i| X.row(i).to_vec())
                .collect(),
        )
        .map_err(|e| {
            DeepBeliefNetworkError::MatrixOperationFailed(format!("Array creation failed: {}", e))
        })?;

        let labeled_y: Vec<i32> = labeled_indices.iter().map(|&i| y[i]).collect();

        // Forward pass through all RBM layers to get final features
        let mut features = labeled_X.clone();
        for rbm in rbm_layers.iter() {
            features = rbm.transform(&features.view())?;
        }

        // Initialize classifier weights
        let feature_dim = features.dim().1;
        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Initialize classifier weights manually
        let mut classifier_weights = Array2::<f64>::zeros((feature_dim, n_classes));
        for i in 0..feature_dim {
            for j in 0..n_classes {
                // Generate normal distributed random number (mean=0.0, std=0.1)
                let u1: f64 = rng.random_range(0.0, 1.0);
                let u2: f64 = rng.random_range(0.0, 1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                classifier_weights[(i, j)] = z * 0.1;
            }
        }
        let mut classifier_bias = Array1::zeros(n_classes);

        // Simple gradient descent for classifier fine-tuning
        let lr = self.learning_rate;
        for epoch in 0..self.finetuning_epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;

            for (sample_idx, &label) in labeled_y.iter().enumerate() {
                let class_idx = unique_classes.iter().position(|&c| c == label).unwrap();
                let feature_vec = features.row(sample_idx);

                // Forward pass
                let mut logits = Array1::zeros(n_classes);
                for j in 0..n_classes {
                    logits[j] = classifier_bias[j] + feature_vec.dot(&classifier_weights.column(j));
                }

                // Softmax
                let max_logit = logits.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_logits: Array1<f64> =
                    logits.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum_exp: f64 = exp_logits.sum();
                let probabilities: Array1<f64> = exp_logits.iter().map(|&x| x / sum_exp).collect();

                // Cross-entropy loss
                let loss = -probabilities[class_idx].ln();
                total_loss += loss;

                // Check prediction
                let predicted_class = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();

                if predicted_class == class_idx {
                    correct_predictions += 1;
                }

                // Backward pass
                let mut target = Array1::zeros(n_classes);
                target[class_idx] = 1.0;
                let error = &probabilities - &target;

                // Update weights and bias
                for j in 0..n_classes {
                    classifier_bias[j] -= lr * error[j];
                    for k in 0..feature_dim {
                        classifier_weights[[k, j]] -= lr * error[j] * feature_vec[k];
                    }
                }
            }

            if epoch % 10 == 0 {
                let accuracy = correct_predictions as f64 / labeled_y.len() as f64;
                println!(
                    "Fine-tuning Epoch {}: Loss = {:.6}, Accuracy = {:.3}",
                    epoch,
                    total_loss / labeled_y.len() as f64,
                    accuracy
                );
            }
        }

        println!("DBN training completed");

        Ok(FittedDeepBeliefNetwork {
            base_model: self.clone(),
            rbm_layers,
            classifier_weights,
            classifier_bias,
            classes: Array1::from_vec(unique_classes),
            n_classes,
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for FittedDeepBeliefNetwork {
    fn predict(&self, X: &ArrayView2<'_, f64>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let n_samples = X.dim().0;
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let predicted_class_idx = probabilities
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            predictions[i] = self.classes[predicted_class_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>> for FittedDeepBeliefNetwork {
    fn predict_proba(&self, X: &ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n_samples = X.dim().0;

        // Forward pass through all RBM layers
        let mut features = X.to_owned();
        for rbm in self.rbm_layers.iter() {
            features = rbm.transform(&features.view())?;
        }

        let mut probabilities = Array2::zeros((n_samples, self.n_classes));

        for i in 0..n_samples {
            let feature_vec = features.row(i);

            // Compute logits
            let mut logits = Array1::zeros(self.n_classes);
            for j in 0..self.n_classes {
                logits[j] =
                    self.classifier_bias[j] + feature_vec.dot(&self.classifier_weights.column(j));
            }

            // Softmax
            let max_logit = logits.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits: Array1<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f64 = exp_logits.sum();

            for j in 0..self.n_classes {
                probabilities[[i, j]] = exp_logits[j] / sum_exp;
            }
        }

        Ok(probabilities)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    fn test_rbm_creation() {
        let rbm = RestrictedBoltzmannMachine::new(10, 5).unwrap();
        assert_eq!(rbm.n_visible, 10);
        assert_eq!(rbm.n_hidden, 5);
        assert_eq!(rbm.learning_rate, 0.01);
        assert_eq!(rbm.n_epochs, 10);
    }

    #[test]
    fn test_rbm_invalid_parameters() {
        assert!(RestrictedBoltzmannMachine::new(0, 5).is_err());
        assert!(RestrictedBoltzmannMachine::new(5, 0).is_err());

        let rbm = RestrictedBoltzmannMachine::new(5, 3).unwrap();
        assert!(rbm.clone().learning_rate(0.0).is_err());
        assert!(rbm.clone().learning_rate(-0.1).is_err());
        assert!(rbm.clone().n_epochs(0).is_err());
        assert!(rbm.clone().batch_size(0).is_err());
        assert!(rbm.clone().n_gibbs_steps(0).is_err());
    }

    #[test]
    fn test_rbm_sigmoid() {
        let rbm = RestrictedBoltzmannMachine::new(3, 2).unwrap();
        assert_abs_diff_eq!(rbm.sigmoid(0.0), 0.5, epsilon = 1e-10);
        assert!(rbm.sigmoid(10.0) > 0.9);
        assert!(rbm.sigmoid(-10.0) < 0.1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rbm_fit_and_transform() {
        let X = array![
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];

        let mut rbm = RestrictedBoltzmannMachine::new(3, 2)
            .unwrap()
            .learning_rate(0.1)
            .unwrap()
            .n_epochs(5)
            .unwrap()
            .batch_size(2)
            .unwrap()
            .random_state(42);

        rbm.fit(&X.view()).unwrap();

        let transformed = rbm.transform(&X.view()).unwrap();
        assert_eq!(transformed.dim(), (4, 2));

        // Check that transformed values are probabilities (between 0 and 1)
        for value in transformed.iter() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rbm_reconstruct() {
        let X = array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];

        let mut rbm = RestrictedBoltzmannMachine::new(3, 2)
            .unwrap()
            .learning_rate(0.1)
            .unwrap()
            .n_epochs(3)
            .unwrap()
            .random_state(42);

        rbm.fit(&X.view()).unwrap();

        let reconstructed = rbm.reconstruct(&X.view()).unwrap();
        assert_eq!(reconstructed.dim(), (2, 3));

        // Check that reconstructed values are binary (0 or 1)
        for value in reconstructed.iter() {
            assert!(*value == 0.0 || *value == 1.0);
        }
    }

    #[test]
    fn test_dbn_creation() {
        let dbn = DeepBeliefNetwork::new()
            .hidden_layers(vec![10, 5])
            .unwrap()
            .learning_rate(0.01)
            .unwrap()
            .pretraining_epochs(5)
            .unwrap()
            .finetuning_epochs(5)
            .unwrap()
            .batch_size(16)
            .unwrap()
            .random_state(42);

        assert_eq!(dbn.hidden_layers, vec![10, 5]);
        assert_eq!(dbn.learning_rate, 0.01);
        assert_eq!(dbn.pretraining_epochs, 5);
        assert_eq!(dbn.finetuning_epochs, 5);
        assert_eq!(dbn.batch_size, 16);
        assert_eq!(dbn.random_state, Some(42));
    }

    #[test]
    fn test_dbn_invalid_parameters() {
        assert!(DeepBeliefNetwork::new().hidden_layers(vec![]).is_err());
        assert!(DeepBeliefNetwork::new().hidden_layers(vec![0, 5]).is_err());
        assert!(DeepBeliefNetwork::new().learning_rate(0.0).is_err());
        assert!(DeepBeliefNetwork::new().pretraining_epochs(0).is_err());
        assert!(DeepBeliefNetwork::new().finetuning_epochs(0).is_err());
        assert!(DeepBeliefNetwork::new().batch_size(0).is_err());
        assert!(DeepBeliefNetwork::new().n_gibbs_steps(0).is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dbn_fit_predict() {
        let X = array![
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0]
        ];
        let y = array![0, 1, 0, 1, -1, -1]; // Last two are unlabeled

        let dbn = DeepBeliefNetwork::new()
            .hidden_layers(vec![3, 2])
            .unwrap()
            .learning_rate(0.1)
            .unwrap()
            .pretraining_epochs(3)
            .unwrap()
            .finetuning_epochs(3)
            .unwrap()
            .batch_size(2)
            .unwrap()
            .random_state(42);

        let fitted = dbn.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 6);

        // Check that predictions are valid class labels
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }

        let probabilities = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probabilities.dim(), (6, 2));

        // Check that probabilities sum to 1
        for i in 0..6 {
            let sum: f64 = probabilities.row(i).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }

        // Check that probabilities are between 0 and 1
        for value in probabilities.iter() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dbn_insufficient_labeled_samples() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![-1, -1]; // All unlabeled

        let dbn = DeepBeliefNetwork::new().hidden_layers(vec![2]).unwrap();

        let result = dbn.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_rbm_hidden_probs_computation() {
        let mut rbm = RestrictedBoltzmannMachine::new(3, 2)
            .unwrap()
            .random_state(42);
        rbm.initialize_weights().unwrap();

        let visible = array![1.0, 0.0, 1.0];
        let hidden_probs = rbm.compute_hidden_probs(&visible.view()).unwrap();

        assert_eq!(hidden_probs.len(), 2);
        for prob in hidden_probs.iter() {
            assert!(*prob >= 0.0 && *prob <= 1.0);
        }
    }
}
