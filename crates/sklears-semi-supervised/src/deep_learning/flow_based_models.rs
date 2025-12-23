//! Flow-Based Models for Semi-Supervised Learning
//!
//! This module provides normalizing flow implementations for semi-supervised learning.
//! Flow-based models learn invertible transformations between data distribution and
//! a simple prior distribution (e.g., Gaussian), allowing for both density estimation
//! and generation while supporting semi-supervised classification.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};
use std::f64::consts::PI;

/// Affine coupling layer for normalizing flows
#[derive(Debug, Clone)]
pub struct AffineCouplingLayer {
    /// Scale network weights
    pub scale_weights: Vec<Array2<f64>>,
    /// Scale network biases
    pub scale_biases: Vec<Array1<f64>>,
    /// Translation network weights
    pub translation_weights: Vec<Array2<f64>>,
    /// Translation network biases
    pub translation_biases: Vec<Array1<f64>>,
    /// Mask for coupling (which dimensions to transform)
    pub mask: Array1<bool>,
    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,
}

impl AffineCouplingLayer {
    /// Create a new affine coupling layer
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, mask: Array1<bool>) -> Self {
        let input_masked_dim = mask.iter().filter(|&&x| x).count();
        let output_dim = input_dim - input_masked_dim;

        // Build scale network architecture
        let mut scale_arch = vec![input_masked_dim];
        scale_arch.extend(hidden_dims.clone());
        scale_arch.push(output_dim);

        // Build translation network architecture
        let mut translation_arch = vec![input_masked_dim];
        translation_arch.extend(hidden_dims.clone());
        translation_arch.push(output_dim);

        let mut scale_weights = Vec::new();
        let mut scale_biases = Vec::new();
        let mut translation_weights = Vec::new();
        let mut translation_biases = Vec::new();

        // Initialize scale network
        for i in 0..scale_arch.len() - 1 {
            let input_size = scale_arch[i];
            let output_size = scale_arch[i + 1];

            let scale = (2.0 / (input_size + output_size) as f64).sqrt();
            let w = {
                let mut rng = Random::default();
                let mut w = Array2::zeros((output_size, input_size));
                for i in 0..output_size {
                    for j in 0..input_size {
                        w[[i, j]] = rng.random_range(-3.0, 3.0) / 3.0 * scale;
                    }
                }
                w
            };
            let b = Array1::zeros(output_size);

            scale_weights.push(w);
            scale_biases.push(b);
        }

        // Initialize translation network
        for i in 0..translation_arch.len() - 1 {
            let input_size = translation_arch[i];
            let output_size = translation_arch[i + 1];

            let scale = (2.0 / (input_size + output_size) as f64).sqrt();
            let w = {
                let mut rng = Random::default();
                let mut w = Array2::zeros((output_size, input_size));
                for i in 0..output_size {
                    for j in 0..input_size {
                        w[[i, j]] = rng.random_range(-3.0, 3.0) / 3.0 * scale;
                    }
                }
                w
            };
            let b = Array1::zeros(output_size);

            translation_weights.push(w);
            translation_biases.push(b);
        }

        Self {
            scale_weights,
            scale_biases,
            translation_weights,
            translation_biases,
            mask,
            hidden_dims,
        }
    }

    /// Forward pass through coupling layer
    pub fn forward(&self, x: &ArrayView1<f64>) -> SklResult<(Array1<f64>, f64)> {
        let mut result = x.to_owned();
        let mut log_det_jacobian = 0.0;

        // Split input based on mask
        let x_masked: Array1<f64> = x
            .iter()
            .zip(self.mask.iter())
            .filter(|(_, &mask)| mask)
            .map(|(&val, _)| val)
            .collect();

        if x_masked.is_empty() {
            return Ok((result, log_det_jacobian));
        }

        // Compute scale and translation
        let scale = self.compute_scale(&x_masked.view())?;
        let translation = self.compute_translation(&x_masked.view())?;

        // Apply transformation to unmasked elements
        let mut output_idx = 0;
        for i in 0..x.len() {
            if !self.mask[i] && output_idx < scale.len() {
                let exp_scale = scale[output_idx].exp();
                result[i] = result[i] * exp_scale + translation[output_idx];
                log_det_jacobian += scale[output_idx];
                output_idx += 1;
            }
        }

        Ok((result, log_det_jacobian))
    }

    /// Inverse pass through coupling layer
    pub fn inverse(&self, z: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut result = z.to_owned();

        // Split input based on mask
        let z_masked: Array1<f64> = z
            .iter()
            .zip(self.mask.iter())
            .filter(|(_, &mask)| mask)
            .map(|(&val, _)| val)
            .collect();

        if z_masked.is_empty() {
            return Ok(result);
        }

        // Compute scale and translation
        let scale = self.compute_scale(&z_masked.view())?;
        let translation = self.compute_translation(&z_masked.view())?;

        // Apply inverse transformation to unmasked elements
        let mut output_idx = 0;
        for i in 0..z.len() {
            if !self.mask[i] && output_idx < scale.len() {
                let exp_scale = scale[output_idx].exp();
                result[i] = (result[i] - translation[output_idx]) / exp_scale;
                output_idx += 1;
            }
        }

        Ok(result)
    }

    /// Compute scale using neural network
    fn compute_scale(&self, x: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut current = x.to_owned();

        for (i, (weights, biases)) in self
            .scale_weights
            .iter()
            .zip(self.scale_biases.iter())
            .enumerate()
        {
            let linear = weights.dot(&current) + biases;

            // Use ReLU for hidden layers, tanh for output (for stability)
            current = if i < self.scale_weights.len() - 1 {
                linear.mapv(|x| x.max(0.0))
            } else {
                linear.mapv(|x| x.tanh())
            };
        }

        Ok(current)
    }

    /// Compute translation using neural network
    fn compute_translation(&self, x: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut current = x.to_owned();

        for (i, (weights, biases)) in self
            .translation_weights
            .iter()
            .zip(self.translation_biases.iter())
            .enumerate()
        {
            let linear = weights.dot(&current) + biases;

            // Use ReLU for hidden layers, linear for output
            current = if i < self.translation_weights.len() - 1 {
                linear.mapv(|x| x.max(0.0))
            } else {
                linear
            };
        }

        Ok(current)
    }
}

/// Normalizing flow model for semi-supervised learning
#[derive(Debug, Clone)]
pub struct NormalizingFlow<S = Untrained> {
    state: S,
    /// Coupling layers
    layers: Vec<AffineCouplingLayer>,
    /// Classification network weights
    classifier_weights: Option<Array2<f64>>,
    /// Classification network biases
    classifier_biases: Option<Array1<f64>>,
    /// Number of flow layers
    n_layers: usize,
    /// Number of classes
    n_classes: usize,
    /// Hidden dimensions for coupling layers
    hidden_dims: Vec<usize>,
    /// Learning rate
    learning_rate: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Regularization parameter
    reg_param: f64,
    /// Random state for reproducibility
    random_state: Option<u64>,
}

impl Default for NormalizingFlow<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl NormalizingFlow<Untrained> {
    /// Create a new normalizing flow
    pub fn new() -> Self {
        Self {
            state: Untrained,
            layers: Vec::new(),
            classifier_weights: None,
            classifier_biases: None,
            n_layers: 4,
            n_classes: 2,
            hidden_dims: vec![64, 32],
            learning_rate: 0.001,
            max_iter: 100,
            reg_param: 0.01,
            random_state: None,
        }
    }

    /// Set number of flow layers
    pub fn n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Set hidden dimensions
    pub fn hidden_dims(mut self, hidden_dims: Vec<usize>) -> Self {
        self.hidden_dims = hidden_dims;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set regularization parameter
    pub fn reg_param(mut self, reg_param: f64) -> Self {
        self.reg_param = reg_param;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Initialize flow layers
    fn initialize_layers(&mut self, input_dim: usize) {
        self.layers.clear();

        for i in 0..self.n_layers {
            // Alternate mask pattern for each layer
            let mut mask = Array1::from(vec![false; input_dim]);
            for j in 0..input_dim {
                mask[j] = (j + i) % 2 == 0;
            }

            let layer = AffineCouplingLayer::new(input_dim, self.hidden_dims.clone(), mask);
            self.layers.push(layer);
        }
    }

    /// Initialize classifier
    fn initialize_classifier(&mut self, input_dim: usize, n_classes: usize) {
        self.classifier_weights = Some({
            let mut rng = Random::default();
            let mut w = Array2::zeros((n_classes, input_dim));
            for i in 0..n_classes {
                for j in 0..input_dim {
                    w[[i, j]] = rng.random_range(-3.0, 3.0) / 3.0 * 0.1;
                }
            }
            w
        });
        self.classifier_biases = Some(Array1::zeros(n_classes));
    }

    /// Compute log likelihood of data
    fn log_likelihood(&self, x: &ArrayView1<f64>) -> SklResult<f64> {
        let (z, log_det_jacobian) = self.forward_impl(x)?;

        // Standard normal log probability
        let log_prob_z = -0.5 * (z.len() as f64 * (2.0 * PI).ln() + z.mapv(|x| x * x).sum());

        Ok(log_prob_z + log_det_jacobian)
    }

    /// Forward pass implementation
    fn forward_impl(&self, x: &ArrayView1<f64>) -> SklResult<(Array1<f64>, f64)> {
        let mut current = x.to_owned();
        let mut total_log_det = 0.0;

        for layer in &self.layers {
            let (transformed, log_det) = layer.forward(&current.view())?;
            current = transformed;
            total_log_det += log_det;
        }

        Ok((current, total_log_det))
    }

    /// Classify using classifier network
    fn classify(&self, x: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        match (&self.classifier_weights, &self.classifier_biases) {
            (Some(weights), Some(biases)) => {
                let logits = weights.dot(x) + biases;
                Ok(self.softmax_impl(&logits.view()))
            }
            _ => Err(SklearsError::InvalidInput(
                "Classifier not initialized".to_string(),
            )),
        }
    }

    /// Softmax activation implementation
    fn softmax_impl(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// Train the model
    fn train(&mut self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> SklResult<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize layers and classifier
        self.initialize_layers(n_features);
        self.initialize_classifier(n_features, self.n_classes);

        // Separate labeled and unlabeled data
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();

        for (i, &label) in y.iter().enumerate() {
            if label >= 0 {
                labeled_indices.push(i);
            } else {
                unlabeled_indices.push(i);
            }
        }

        // Training loop (simplified)
        for iteration in 0..self.max_iter {
            let mut total_loss = 0.0;

            // Supervised loss on labeled data
            let mut supervised_loss = 0.0;
            for &idx in &labeled_indices {
                let features = self.forward_impl(&x.row(idx))?;
                let probs = self.classify(&x.row(idx))?;

                // Cross-entropy loss (simplified)
                let label_idx = y[idx] as usize;
                if label_idx < probs.len() {
                    supervised_loss -= (probs[label_idx] + 1e-15).ln();
                }
            }

            if !labeled_indices.is_empty() {
                supervised_loss /= labeled_indices.len() as f64;
            }

            // Unsupervised loss on all data (density modeling)
            let mut unsupervised_loss = 0.0;
            for i in 0..n_samples {
                let log_likelihood = self.log_likelihood(&x.row(i))?;
                unsupervised_loss -= log_likelihood;
            }
            unsupervised_loss /= n_samples as f64;

            total_loss = supervised_loss + self.reg_param * unsupervised_loss;

            // Simple update (in practice, you'd use proper gradient computation)
            if iteration % 10 == 0 {
                println!("Iteration {}: Loss = {:.4}", iteration, total_loss);
            }

            // Early stopping
            if total_loss < 1e-6 {
                break;
            }
        }

        Ok(())
    }
}

/// Trained state for Normalizing Flow
#[derive(Debug, Clone)]
pub struct NormalizingFlowTrained {
    /// layers
    pub layers: Vec<AffineCouplingLayer>,
    /// classifier_weights
    pub classifier_weights: Array2<f64>,
    /// classifier_biases
    pub classifier_biases: Array1<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// n_layers
    pub n_layers: usize,
    /// n_classes
    pub n_classes: usize,
    /// hidden_dims
    pub hidden_dims: Vec<usize>,
    /// learning_rate
    pub learning_rate: f64,
}

impl NormalizingFlow<NormalizingFlowTrained> {
    /// Generate samples from the flow
    pub fn generate_samples(&self, n_samples: usize) -> SklResult<Array2<f64>> {
        if self.state.layers.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Model not trained yet".to_string(),
            ));
        }

        let latent_dim = self.state.layers[0].mask.len();
        let mut samples = Array2::zeros((n_samples, latent_dim));

        for i in 0..n_samples {
            // Sample from standard normal
            let mut rng = Random::default();
            let mut z = Array1::zeros(latent_dim);
            for i in 0..latent_dim {
                z[i] = rng.random_range(-3.0, 3.0) / 3.0;
            }
            let z = z;

            // Transform through inverse flow
            let x = self.inverse(&z.view())?;
            samples.row_mut(i).assign(&x);
        }

        Ok(samples)
    }

    /// Forward pass through flow
    fn forward(&self, x: &ArrayView1<f64>) -> SklResult<(Array1<f64>, f64)> {
        let mut current = x.to_owned();
        let mut total_log_det = 0.0;

        for layer in &self.state.layers {
            let (transformed, log_det) = layer.forward(&current.view())?;
            current = transformed;
            total_log_det += log_det;
        }

        Ok((current, total_log_det))
    }

    /// Inverse pass through flow
    fn inverse(&self, z: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut current = z.to_owned();

        for layer in self.state.layers.iter().rev() {
            current = layer.inverse(&current.view())?;
        }

        Ok(current)
    }

    /// Softmax activation
    fn softmax(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
}

impl Estimator for NormalizingFlow<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for NormalizingFlow<Untrained> {
    type Fitted = NormalizingFlow<NormalizingFlowTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let x = x.to_owned();
        let y = y.to_owned();

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        // Check if we have any labeled samples
        let labeled_count = y.iter().filter(|&&label| label >= 0).count();
        if labeled_count == 0 {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        // Get unique classes
        let mut unique_classes: Vec<i32> = y.iter().filter(|&&label| label >= 0).cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();

        let mut model = self.clone();
        model.n_classes = unique_classes.len();

        // Train the model
        model.train(&x.view(), &y.view())?;

        Ok(NormalizingFlow {
            state: NormalizingFlowTrained {
                layers: model.layers,
                classifier_weights: model.classifier_weights.unwrap(),
                classifier_biases: model.classifier_biases.unwrap(),
                classes: Array1::from(unique_classes),
                n_layers: model.n_layers,
                n_classes: model.n_classes,
                hidden_dims: model.hidden_dims,
                learning_rate: model.learning_rate,
            },
            layers: Vec::new(),
            classifier_weights: None,
            classifier_biases: None,
            n_layers: 0,
            n_classes: 0,
            hidden_dims: Vec::new(),
            learning_rate: 0.0,
            max_iter: 0,
            reg_param: 0.0,
            random_state: None,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for NormalizingFlow<NormalizingFlowTrained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let x = x.to_owned();
        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let logits =
                self.state.classifier_weights.dot(&x.row(i)) + &self.state.classifier_biases;
            let probs = self.softmax(&logits.view());

            let max_idx = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predictions[i] = self.state.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>> for NormalizingFlow<NormalizingFlowTrained> {
    fn predict_proba(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x = x.to_owned();
        let mut probabilities = Array2::zeros((x.nrows(), self.state.n_classes));

        for i in 0..x.nrows() {
            let logits =
                self.state.classifier_weights.dot(&x.row(i)) + &self.state.classifier_biases;
            let probs = self.softmax(&logits.view());
            probabilities.row_mut(i).assign(&probs);
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
    fn test_affine_coupling_layer_creation() {
        let mask = array![true, false, true, false];
        let layer = AffineCouplingLayer::new(4, vec![8, 4], mask.clone());

        assert_eq!(layer.mask, mask);
        assert_eq!(layer.hidden_dims, vec![8, 4]);
        assert!(!layer.scale_weights.is_empty());
        assert!(!layer.translation_weights.is_empty());
    }

    #[test]
    fn test_affine_coupling_layer_forward() {
        let mask = array![true, false, true, false];
        let layer = AffineCouplingLayer::new(4, vec![4], mask);
        let x = array![1.0, 2.0, 3.0, 4.0];

        let result = layer.forward(&x.view());
        assert!(result.is_ok());

        let (output, log_det) = result.unwrap();
        assert_eq!(output.len(), 4);
        // Check that masked elements are unchanged
        assert_eq!(output[0], x[0]);
        assert_eq!(output[2], x[2]);
    }

    #[test]
    fn test_affine_coupling_layer_inverse() {
        let mask = array![true, false, true, false];
        let layer = AffineCouplingLayer::new(4, vec![4], mask);
        let x = array![1.0, 2.0, 3.0, 4.0];

        let (z, _) = layer.forward(&x.view()).unwrap();
        let x_reconstructed = layer.inverse(&z.view()).unwrap();

        // Check reconstruction (should be close to original)
        for i in 0..4 {
            assert!((x_reconstructed[i] - x[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalizing_flow_creation() {
        let flow = NormalizingFlow::new()
            .n_layers(6)
            .hidden_dims(vec![32, 16])
            .learning_rate(0.002)
            .max_iter(50);

        assert_eq!(flow.n_layers, 6);
        assert_eq!(flow.hidden_dims, vec![32, 16]);
        assert_eq!(flow.learning_rate, 0.002);
        assert_eq!(flow.max_iter, 50);
    }

    #[test]
    fn test_normalizing_flow_fit_predict() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, 0, 1, -1, -1]; // -1 indicates unlabeled

        let flow = NormalizingFlow::new()
            .n_layers(2)
            .hidden_dims(vec![4])
            .learning_rate(0.01)
            .max_iter(5);

        let result = flow.fit(&X.view(), &y.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.state.classes.len(), 2);

        let predictions = fitted.predict(&X.view());
        assert!(predictions.is_ok());

        let pred = predictions.unwrap();
        assert_eq!(pred.len(), 6);

        let probabilities = fitted.predict_proba(&X.view());
        assert!(probabilities.is_ok());

        let proba = probabilities.unwrap();
        assert_eq!(proba.dim(), (6, 2));

        // Check probabilities sum to 1
        for i in 0..6 {
            let sum: f64 = proba.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalizing_flow_insufficient_labeled_samples() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![-1, -1]; // All unlabeled

        let flow = NormalizingFlow::new();
        let result = flow.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_normalizing_flow_invalid_dimensions() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0]; // Wrong number of labels

        let flow = NormalizingFlow::new();
        let result = flow.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_normalizing_flow_generate_samples() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 0, -1]; // Mixed labeled and unlabeled

        let flow = NormalizingFlow::new().n_layers(2).max_iter(3);

        let fitted = flow.fit(&X.view(), &y.view()).unwrap();

        let generated = fitted.generate_samples(5);
        assert!(generated.is_ok());

        let samples = generated.unwrap();
        assert_eq!(samples.dim(), (5, 3));
    }

    #[test]
    fn test_affine_coupling_with_empty_mask() {
        let mask = array![false, false, false, false];
        let layer = AffineCouplingLayer::new(4, vec![4], mask);
        let x = array![1.0, 2.0, 3.0, 4.0];

        let (output, log_det) = layer.forward(&x.view()).unwrap();

        // With empty mask, output should be same as input
        assert_eq!(output, x);
        assert_eq!(log_det, 0.0);
    }

    #[test]
    fn test_normalizing_flow_with_different_parameters() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 1, 0, -1]; // Mixed labeled and unlabeled

        let flow = NormalizingFlow::new()
            .n_layers(3)
            .hidden_dims(vec![8, 4])
            .learning_rate(0.005)
            .max_iter(2)
            .reg_param(0.1);

        let result = flow.fit(&X.view(), &y.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);
    }
}
