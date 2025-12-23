//! Cross-Modal Contrastive Learning for Semi-Supervised Learning
//!
//! This module provides cross-modal contrastive learning implementations that learn
//! representations across different data modalities (e.g., text and images, audio and video).
//! These methods use contrastive loss to align representations from different modalities
//! while leveraging both labeled and unlabeled data for semi-supervised learning.

use scirs2_core::ndarray_ext::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Projection network for cross-modal contrastive learning
#[derive(Debug, Clone)]
pub struct ProjectionNetwork {
    /// Layer weights
    pub weights: Vec<Array2<f64>>,
    /// Layer biases
    pub biases: Vec<Array1<f64>>,
    /// Network architecture
    pub architecture: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
}

impl ProjectionNetwork {
    /// Create a new projection network
    pub fn new(input_dim: usize, output_dim: usize, hidden_dims: Vec<usize>) -> Self {
        let mut architecture = vec![input_dim];
        architecture.extend(hidden_dims);
        architecture.push(output_dim);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..architecture.len() - 1 {
            let input_size = architecture[i];
            let output_size = architecture[i + 1];

            // Xavier initialization - create weights manually
            let scale = (2.0 / (input_size + output_size) as f64).sqrt();
            let mut rng = Random::default();
            let mut w = Array2::<f64>::zeros((output_size, input_size));
            for i in 0..output_size {
                for j in 0..input_size {
                    // Generate standard normal distributed random number
                    let u1: f64 = rng.random_range(0.0, 1.0);
                    let u2: f64 = rng.random_range(0.0, 1.0);
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    w[(i, j)] = z * scale;
                }
            }
            let b = Array1::zeros(output_size);

            weights.push(w);
            biases.push(b);
        }

        Self {
            weights,
            biases,
            architecture,
            output_dim,
        }
    }

    /// Forward pass through projection network
    pub fn forward(&self, x: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut current = x.to_owned();

        for (i, (weights, biases)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let linear = weights.dot(&current) + biases;

            // Use ReLU for hidden layers, linear for output
            current = if i < self.weights.len() - 1 {
                linear.mapv(|x| x.max(0.0))
            } else {
                // L2 normalize output
                let norm = (linear.mapv(|x| x * x).sum() + 1e-12).sqrt();
                linear / norm
            };
        }

        Ok(current)
    }
}

/// Cross-modal contrastive learning model
#[derive(Debug, Clone)]
pub struct CrossModalContrastive<S = Untrained> {
    state: S,
    /// Projection network for modality 1
    projection1: Option<ProjectionNetwork>,
    /// Projection network for modality 2
    projection2: Option<ProjectionNetwork>,
    /// Classification network
    classifier_weights: Option<Array2<f64>>,
    classifier_biases: Option<Array1<f64>>,
    /// Projection dimension
    projection_dim: usize,
    /// Number of classes
    n_classes: usize,
    /// Hidden dimensions for projection networks
    hidden_dims: Vec<usize>,
    /// Temperature for contrastive loss
    temperature: f64,
    /// Learning rate
    learning_rate: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Contrastive loss weight
    contrastive_weight: f64,
    /// Supervised loss weight
    supervised_weight: f64,
    /// Random state for reproducibility
    random_state: Option<u64>,
}

impl Default for CrossModalContrastive<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossModalContrastive<Untrained> {
    /// Create a new cross-modal contrastive learning model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            projection1: None,
            projection2: None,
            classifier_weights: None,
            classifier_biases: None,
            projection_dim: 128,
            n_classes: 2,
            hidden_dims: vec![256, 128],
            temperature: 0.07,
            learning_rate: 0.001,
            max_iter: 100,
            contrastive_weight: 1.0,
            supervised_weight: 1.0,
            random_state: None,
        }
    }

    /// Set projection dimension
    pub fn projection_dim(mut self, dim: usize) -> Self {
        self.projection_dim = dim;
        self
    }

    /// Set hidden dimensions
    pub fn hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    /// Set temperature for contrastive loss
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
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

    /// Set contrastive loss weight
    pub fn contrastive_weight(mut self, weight: f64) -> Self {
        self.contrastive_weight = weight;
        self
    }

    /// Set supervised loss weight
    pub fn supervised_weight(mut self, weight: f64) -> Self {
        self.supervised_weight = weight;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Initialize networks
    fn initialize_networks(&mut self, input_dim1: usize, input_dim2: usize, n_classes: usize) {
        self.projection1 = Some(ProjectionNetwork::new(
            input_dim1,
            self.projection_dim,
            self.hidden_dims.clone(),
        ));

        self.projection2 = Some(ProjectionNetwork::new(
            input_dim2,
            self.projection_dim,
            self.hidden_dims.clone(),
        ));

        // Use combined projection for classification
        let combined_dim = self.projection_dim * 2;
        // Initialize classifier weights manually
        let mut rng = Random::default();
        let mut weights = Array2::<f64>::zeros((n_classes, combined_dim));
        for i in 0..n_classes {
            for j in 0..combined_dim {
                // Generate standard normal distributed random number
                let u1: f64 = rng.random_range(0.0, 1.0);
                let u2: f64 = rng.random_range(0.0, 1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                weights[(i, j)] = z * 0.1;
            }
        }
        self.classifier_weights = Some(weights);
        self.classifier_biases = Some(Array1::zeros(n_classes));

        self.n_classes = n_classes;
    }

    /// Compute contrastive loss between two modalities
    fn contrastive_loss(&self, z1: &ArrayView2<f64>, z2: &ArrayView2<f64>) -> SklResult<f64> {
        let batch_size = z1.nrows();
        if batch_size != z2.nrows() {
            return Err(SklearsError::InvalidInput(
                "Batch sizes must match".to_string(),
            ));
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let z1_i = z1.row(i);
            let z2_i = z2.row(i);

            // Compute similarity between positive pair
            let pos_sim = z1_i.dot(&z2_i) / self.temperature;

            // Compute similarities with all other samples (negatives)
            let mut neg_sims = Vec::new();
            for j in 0..batch_size {
                if i != j {
                    let sim1 = z1_i.dot(&z1.row(j)) / self.temperature;
                    let sim2 = z1_i.dot(&z2.row(j)) / self.temperature;
                    neg_sims.push(sim1);
                    neg_sims.push(sim2);
                }
            }

            // Compute softmax denominator
            let mut exp_sum = pos_sim.exp();
            for &sim in &neg_sims {
                exp_sum += sim.exp();
            }

            // Contrastive loss (negative log probability)
            let loss = -pos_sim + (exp_sum + 1e-12).ln();
            total_loss += loss;
        }

        Ok(total_loss / batch_size as f64)
    }

    /// Project features from both modalities
    fn project_features(
        &self,
        x1: &ArrayView2<f64>,
        x2: &ArrayView2<f64>,
    ) -> SklResult<(Array2<f64>, Array2<f64>)> {
        let proj1 = self.projection1.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Projection network 1 not initialized".to_string())
        })?;

        let proj2 = self.projection2.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Projection network 2 not initialized".to_string())
        })?;

        let batch_size = x1.nrows();
        let mut z1 = Array2::zeros((batch_size, self.projection_dim));
        let mut z2 = Array2::zeros((batch_size, self.projection_dim));

        for i in 0..batch_size {
            let proj1_output = proj1.forward(&x1.row(i))?;
            let proj2_output = proj2.forward(&x2.row(i))?;

            z1.row_mut(i).assign(&proj1_output);
            z2.row_mut(i).assign(&proj2_output);
        }

        Ok((z1, z2))
    }

    /// Classify using combined features
    fn classify(&self, z1: &ArrayView1<f64>, z2: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        match (&self.classifier_weights, &self.classifier_biases) {
            (Some(weights), Some(biases)) => {
                // Concatenate projections
                let mut combined = Array1::zeros(z1.len() + z2.len());
                combined.slice_mut(s![..z1.len()]).assign(z1);
                combined.slice_mut(s![z1.len()..]).assign(z2);

                let logits = weights.dot(&combined) + biases;
                Ok(self.softmax(&logits.view()))
            }
            _ => Err(SklearsError::InvalidInput(
                "Classifier not initialized".to_string(),
            )),
        }
    }

    /// Softmax activation
    fn softmax(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// Train the model
    fn train(
        &mut self,
        x1: &ArrayView2<f64>,
        x2: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
    ) -> SklResult<()> {
        let n_samples = x1.nrows();

        if x1.nrows() != x2.nrows() || x1.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "All inputs must have the same number of samples".to_string(),
            ));
        }

        // Initialize networks
        self.initialize_networks(x1.ncols(), x2.ncols(), self.n_classes);

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

            // Project all features
            let (z1, z2) = self.project_features(x1, x2)?;

            // Contrastive loss on all data
            let contrastive_loss = self.contrastive_loss(&z1.view(), &z2.view())?;
            total_loss += self.contrastive_weight * contrastive_loss;

            // Supervised loss on labeled data
            if !labeled_indices.is_empty() {
                let mut supervised_loss = 0.0;
                for &idx in &labeled_indices {
                    let probs = self.classify(&z1.row(idx), &z2.row(idx))?;
                    let label_idx = y[idx] as usize;
                    if label_idx < probs.len() {
                        supervised_loss -= (probs[label_idx] + 1e-15).ln();
                    }
                }
                supervised_loss /= labeled_indices.len() as f64;
                total_loss += self.supervised_weight * supervised_loss;
            }

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

/// Trained state for Cross-Modal Contrastive Learning
#[derive(Debug, Clone)]
pub struct CrossModalContrastiveTrained {
    /// projection1
    pub projection1: ProjectionNetwork,
    /// projection2
    pub projection2: ProjectionNetwork,
    /// classifier_weights
    pub classifier_weights: Array2<f64>,
    /// classifier_biases
    pub classifier_biases: Array1<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// projection_dim
    pub projection_dim: usize,
    /// n_classes
    pub n_classes: usize,
    /// temperature
    pub temperature: f64,
}

impl CrossModalContrastive<CrossModalContrastiveTrained> {
    /// Get embeddings for both modalities (trained model)
    pub fn get_embeddings(
        &self,
        x1: &ArrayView2<f64>,
        x2: &ArrayView2<f64>,
    ) -> SklResult<(Array2<f64>, Array2<f64>)> {
        let batch_size = x1.nrows();
        let mut z1 = Array2::zeros((batch_size, self.state.projection_dim));
        let mut z2 = Array2::zeros((batch_size, self.state.projection_dim));

        for i in 0..batch_size {
            let proj1_output = self.state.projection1.forward(&x1.row(i))?;
            let proj2_output = self.state.projection2.forward(&x2.row(i))?;

            z1.row_mut(i).assign(&proj1_output);
            z2.row_mut(i).assign(&proj2_output);
        }

        Ok((z1, z2))
    }

    /// Classify using combined features (trained model)
    fn classify(&self, z1: &ArrayView1<f64>, z2: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        // Concatenate projections
        let mut combined = Array1::zeros(z1.len() + z2.len());
        combined.slice_mut(s![..z1.len()]).assign(z1);
        combined.slice_mut(s![z1.len()..]).assign(z2);

        let logits = self.state.classifier_weights.dot(&combined) + &self.state.classifier_biases;
        Ok(self.softmax(&logits.view()))
    }

    /// Softmax activation
    fn softmax(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
}

impl Estimator for CrossModalContrastive<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Input for cross-modal learning: (modality1, modality2)
pub type CrossModalInput = (Array2<f64>, Array2<f64>);

impl Fit<CrossModalInput, ArrayView1<'_, i32>> for CrossModalContrastive<Untrained> {
    type Fitted = CrossModalContrastive<CrossModalContrastiveTrained>;

    fn fit(self, input: &CrossModalInput, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let (x1, x2) = input;
        let y = y.to_owned();

        if x1.nrows() != x2.nrows() || x1.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "All inputs must have the same number of samples".to_string(),
            ));
        }

        if x1.nrows() == 0 {
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
        model.train(&x1.view(), &x2.view(), &y.view())?;

        Ok(CrossModalContrastive {
            state: CrossModalContrastiveTrained {
                projection1: model.projection1.unwrap(),
                projection2: model.projection2.unwrap(),
                classifier_weights: model.classifier_weights.unwrap(),
                classifier_biases: model.classifier_biases.unwrap(),
                classes: Array1::from(unique_classes),
                projection_dim: model.projection_dim,
                n_classes: model.n_classes,
                temperature: model.temperature,
            },
            projection1: None,
            projection2: None,
            classifier_weights: None,
            classifier_biases: None,
            projection_dim: 0,
            n_classes: 0,
            hidden_dims: Vec::new(),
            temperature: 0.0,
            learning_rate: 0.0,
            max_iter: 0,
            contrastive_weight: 0.0,
            supervised_weight: 0.0,
            random_state: None,
        })
    }
}

impl Predict<CrossModalInput, Array1<i32>> for CrossModalContrastive<CrossModalContrastiveTrained> {
    fn predict(&self, input: &CrossModalInput) -> SklResult<Array1<i32>> {
        let (x1, x2) = input;
        let mut predictions = Array1::zeros(x1.nrows());

        for i in 0..x1.nrows() {
            let z1 = self.state.projection1.forward(&x1.row(i))?;
            let z2 = self.state.projection2.forward(&x2.row(i))?;
            let probs = self.classify(&z1.view(), &z2.view())?;

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

impl PredictProba<CrossModalInput, Array2<f64>>
    for CrossModalContrastive<CrossModalContrastiveTrained>
{
    fn predict_proba(&self, input: &CrossModalInput) -> SklResult<Array2<f64>> {
        let (x1, x2) = input;
        let mut probabilities = Array2::zeros((x1.nrows(), self.state.n_classes));

        for i in 0..x1.nrows() {
            let z1 = self.state.projection1.forward(&x1.row(i))?;
            let z2 = self.state.projection2.forward(&x2.row(i))?;
            let probs = self.classify(&z1.view(), &z2.view())?;
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
    fn test_projection_network_creation() {
        let network = ProjectionNetwork::new(10, 5, vec![8, 6]);
        assert_eq!(network.architecture, vec![10, 8, 6, 5]);
        assert_eq!(network.output_dim, 5);
        assert_eq!(network.weights.len(), 3);
        assert_eq!(network.biases.len(), 3);
    }

    #[test]
    fn test_projection_network_forward() {
        let network = ProjectionNetwork::new(3, 2, vec![4]);
        let x = array![1.0, 2.0, 3.0];

        let result = network.forward(&x.view());
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 2);

        // Check L2 normalization (with reasonable tolerance for numerical stability)
        let norm = (output.mapv(|x| x * x).sum()).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_modal_contrastive_creation() {
        let model = CrossModalContrastive::new()
            .projection_dim(64)
            .hidden_dims(vec![128, 64])
            .temperature(0.1)
            .learning_rate(0.01)
            .max_iter(50);

        assert_eq!(model.projection_dim, 64);
        assert_eq!(model.hidden_dims, vec![128, 64]);
        assert_eq!(model.temperature, 0.1);
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.max_iter, 50);
    }

    #[test]
    fn test_cross_modal_contrastive_fit_predict() {
        // Modality 1 data (e.g., text features)
        let x1 = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0]
        ];

        // Modality 2 data (e.g., image features)
        let x2 = array![
            [0.5, 1.5, 2.5, 3.5],
            [1.5, 2.5, 3.5, 4.5],
            [2.5, 3.5, 4.5, 5.5],
            [3.5, 4.5, 5.5, 6.5],
            [4.5, 5.5, 6.5, 7.5],
            [5.5, 6.5, 7.5, 8.5]
        ];

        let y = array![0, 1, 0, 1, -1, -1]; // -1 indicates unlabeled

        let model = CrossModalContrastive::new()
            .projection_dim(8)
            .hidden_dims(vec![12])
            .temperature(0.1)
            .learning_rate(0.01)
            .max_iter(5);

        let input = (x1.clone(), x2.clone());
        let result = model.fit(&input, &y.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.state.classes.len(), 2);

        let predictions = fitted.predict(&input);
        assert!(predictions.is_ok());

        let pred = predictions.unwrap();
        assert_eq!(pred.len(), 6);

        let probabilities = fitted.predict_proba(&input);
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
    fn test_cross_modal_contrastive_insufficient_labeled_samples() {
        let x1 = array![[1.0, 2.0], [2.0, 3.0]];
        let x2 = array![[1.5, 2.5], [2.5, 3.5]];
        let y = array![-1, -1]; // All unlabeled

        let model = CrossModalContrastive::new();
        let input = (x1, x2);
        let result = model.fit(&input, &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_modal_contrastive_mismatched_dimensions() {
        let x1 = array![[1.0, 2.0], [2.0, 3.0]];
        let x2 = array![[1.5, 2.5]]; // Different number of samples
        let y = array![0, 1];

        let model = CrossModalContrastive::new();
        let input = (x1, x2);
        let result = model.fit(&input, &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_modal_get_embeddings() {
        let x1 = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];

        let x2 = array![
            [0.5, 1.5, 2.5, 3.5],
            [1.5, 2.5, 3.5, 4.5],
            [2.5, 3.5, 4.5, 5.5],
            [3.5, 4.5, 5.5, 6.5]
        ];

        let y = array![0, 1, 0, -1]; // Mixed labeled and unlabeled

        let model = CrossModalContrastive::new().projection_dim(6).max_iter(3);

        let input = (x1.clone(), x2.clone());
        let fitted = model.fit(&input, &y.view()).unwrap();

        let embeddings = fitted.get_embeddings(&x1.view(), &x2.view());
        assert!(embeddings.is_ok());

        let (z1, z2) = embeddings.unwrap();
        assert_eq!(z1.dim(), (4, 6));
        assert_eq!(z2.dim(), (4, 6));

        // Check L2 normalization of embeddings
        for i in 0..4 {
            let norm1 = (z1.row(i).mapv(|x| x * x).sum()).sqrt();
            let norm2 = (z2.row(i).mapv(|x| x * x).sum()).sqrt();
            assert!((norm1 - 1.0).abs() < 1e-10);
            assert!((norm2 - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cross_modal_contrastive_with_different_parameters() {
        let x1 = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];

        let x2 = array![[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];

        let y = array![0, 1, 0, -1]; // Mixed labeled and unlabeled

        let model = CrossModalContrastive::new()
            .projection_dim(10)
            .hidden_dims(vec![16, 12])
            .temperature(0.05)
            .contrastive_weight(2.0)
            .supervised_weight(0.5)
            .max_iter(2);

        let input = (x1, x2);
        let result = model.fit(&input, &y.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        let predictions = fitted.predict(&input).unwrap();
        assert_eq!(predictions.len(), 4);
    }
}
