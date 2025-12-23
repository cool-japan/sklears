//! Stacked Autoencoders for semi-supervised learning
//!
//! This module implements stacked autoencoders, which are deep neural networks
//! composed of multiple autoencoder layers. They can be used for unsupervised
//! pre-training followed by supervised fine-tuning for semi-supervised learning.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Random, Rng};
// use scirs2_core::random::rand::seq::SliceRandom;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict, Transform};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StackedAutoencoderError {
    #[error("Invalid layer sizes: {0:?}")]
    InvalidLayerSizes(Vec<usize>),
    #[error("Invalid learning rate: {0}")]
    InvalidLearningRate(f64),
    #[error("Invalid epochs: {0}")]
    InvalidEpochs(usize),
    #[error("Insufficient labeled samples: need at least 1")]
    InsufficientLabeledSamples,
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    #[error("Model not trained")]
    ModelNotTrained,
}

impl From<StackedAutoencoderError> for SklearsError {
    fn from(err: StackedAutoencoderError) -> Self {
        SklearsError::FitError(err.to_string())
    }
}

/// Single Autoencoder layer for pre-training
#[derive(Debug, Clone)]
pub struct AutoencoderLayer {
    /// input_size
    pub input_size: usize,
    /// hidden_size
    pub hidden_size: usize,
    /// learning_rate
    pub learning_rate: f64,
    /// epochs
    pub epochs: usize,
    /// noise_factor
    pub noise_factor: f64,
    weights_encode: Array2<f64>,
    bias_encode: Array1<f64>,
    weights_decode: Array2<f64>,
    bias_decode: Array1<f64>,
    is_trained: bool,
}

impl AutoencoderLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            learning_rate: 0.001,
            epochs: 100,
            noise_factor: 0.1,
            weights_encode: Array2::zeros((input_size, hidden_size)),
            bias_encode: Array1::zeros(hidden_size),
            weights_decode: Array2::zeros((hidden_size, input_size)),
            bias_decode: Array1::zeros(input_size),
            is_trained: false,
        }
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(StackedAutoencoderError::InvalidLearningRate(learning_rate).into());
        }
        self.learning_rate = learning_rate;
        Ok(self)
    }

    pub fn epochs(mut self, epochs: usize) -> Result<Self> {
        if epochs == 0 {
            return Err(StackedAutoencoderError::InvalidEpochs(epochs).into());
        }
        self.epochs = epochs;
        Ok(self)
    }

    pub fn noise_factor(mut self, noise_factor: f64) -> Self {
        self.noise_factor = noise_factor;
        self
    }

    fn initialize_weights(&mut self, random_state: Option<u64>) {
        let mut rng = match random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Xavier initialization
        let limit_encode = (6.0 / (self.input_size + self.hidden_size) as f64).sqrt();
        let limit_decode = (6.0 / (self.hidden_size + self.input_size) as f64).sqrt();

        // Initialize encoder weights manually
        let mut weights_encode = Array2::<f64>::zeros((self.input_size, self.hidden_size));
        for i in 0..self.input_size {
            for j in 0..self.hidden_size {
                // Generate uniform distributed random number in [-limit_encode, limit_encode]
                let u: f64 = rng.random_range(0.0, 1.0);
                weights_encode[(i, j)] = u * (2.0 * limit_encode) - limit_encode;
            }
        }
        self.weights_encode = weights_encode;

        // Initialize decoder weights manually
        let mut weights_decode = Array2::<f64>::zeros((self.hidden_size, self.input_size));
        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                // Generate uniform distributed random number in [-limit_decode, limit_decode]
                let u: f64 = rng.random_range(0.0, 1.0);
                weights_decode[(i, j)] = u * (2.0 * limit_decode) - limit_decode;
            }
        }
        self.weights_decode = weights_decode;

        self.bias_encode = Array1::zeros(self.hidden_size);
        self.bias_decode = Array1::zeros(self.input_size);
    }

    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(&self, x: f64) -> f64 {
        let s = self.sigmoid(x);
        s * (1.0 - s)
    }

    fn add_noise<R>(&self, X: &ArrayView2<f64>, rng: &mut Random<R>) -> Array2<f64>
    where
        R: Rng,
    {
        // Generate noise manually
        let (nrows, ncols) = X.dim();
        let mut noise = Array2::<f64>::zeros((nrows, ncols));
        for i in 0..nrows {
            for j in 0..ncols {
                // Generate normal distributed random number (mean=0.0, std=noise_factor)
                let u1: f64 = rng.random_range(0.0, 1.0);
                let u2: f64 = rng.random_range(0.0, 1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                noise[(i, j)] = z * self.noise_factor;
            }
        }
        X + &noise
    }

    fn encode(&self, X: &ArrayView2<f64>) -> Array2<f64> {
        let linear = X.dot(&self.weights_encode) + &self.bias_encode;
        linear.mapv(|x| self.sigmoid(x))
    }

    fn decode(&self, H: &ArrayView2<f64>) -> Array2<f64> {
        let linear = H.dot(&self.weights_decode) + &self.bias_decode;
        linear.mapv(|x| self.sigmoid(x))
    }

    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &ArrayView2<f64>, random_state: Option<u64>) -> Result<()> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.input_size {
            return Err(StackedAutoencoderError::ShapeMismatch {
                expected: vec![n_samples, self.input_size],
                actual: vec![n_samples, n_features],
            }
            .into());
        }

        self.initialize_weights(random_state);

        let mut rng = match random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Training loop
        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;

            // Add noise for denoising autoencoder
            let X_noisy = self.add_noise(X, &mut rng);

            // Forward pass
            let encoded = self.encode(&X_noisy.view());
            let decoded = self.decode(&encoded.view());

            // Compute reconstruction loss (MSE)
            let reconstruction_error = X - &decoded;
            total_loss = reconstruction_error.mapv(|x| x * x).sum() / n_samples as f64;

            // Backward pass
            // Output layer gradients
            let output_delta = &reconstruction_error * 2.0 / n_samples as f64;

            // Hidden layer gradients
            let hidden_linear = X_noisy.dot(&self.weights_encode) + &self.bias_encode;
            let hidden_delta = output_delta.dot(&self.weights_decode.t())
                * hidden_linear.mapv(|x| self.sigmoid_derivative(x));

            // Update weights and biases
            let dW_decode = encoded.t().dot(&output_delta);
            let db_decode = output_delta.sum_axis(Axis(0));

            let dW_encode = X_noisy.t().dot(&hidden_delta);
            let db_encode = hidden_delta.sum_axis(Axis(0));

            self.weights_decode = &self.weights_decode - self.learning_rate * dW_decode;
            self.bias_decode = &self.bias_decode - self.learning_rate * db_decode;

            self.weights_encode = &self.weights_encode - self.learning_rate * dW_encode;
            self.bias_encode = &self.bias_encode - self.learning_rate * db_encode;

            // Early stopping check
            if epoch % 10 == 0 && total_loss < 1e-6 {
                break;
            }
        }

        self.is_trained = true;
        Ok(())
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if !self.is_trained {
            return Err(StackedAutoencoderError::ModelNotTrained.into());
        }

        let (_, n_features) = X.dim();
        if n_features != self.input_size {
            return Err(StackedAutoencoderError::ShapeMismatch {
                expected: vec![0, self.input_size],
                actual: vec![0, n_features],
            }
            .into());
        }

        Ok(self.encode(X))
    }

    pub fn reconstruct(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if !self.is_trained {
            return Err(StackedAutoencoderError::ModelNotTrained.into());
        }

        let encoded = self.transform(X)?;
        Ok(self.decode(&encoded.view()))
    }
}

/// Stacked Autoencoders for semi-supervised learning
///
/// This implements a deep architecture of stacked autoencoders that can be
/// pre-trained in an unsupervised manner and then fine-tuned for classification.
#[derive(Debug, Clone)]
pub struct StackedAutoencoders {
    /// layer_sizes
    pub layer_sizes: Vec<usize>,
    /// learning_rate
    pub learning_rate: f64,
    /// pretrain_epochs
    pub pretrain_epochs: usize,
    /// finetune_epochs
    pub finetune_epochs: usize,
    /// noise_factor
    pub noise_factor: f64,
    /// random_state
    pub random_state: Option<u64>,
    layers: Vec<AutoencoderLayer>,
    classifier_weights: Array2<f64>,
    classifier_bias: Array1<f64>,
    n_classes: usize,
    is_pretrained: bool,
    is_finetuned: bool,
}

impl Default for StackedAutoencoders {
    fn default() -> Self {
        Self {
            layer_sizes: vec![784, 500, 200, 50],
            learning_rate: 0.001,
            pretrain_epochs: 100,
            finetune_epochs: 50,
            noise_factor: 0.1,
            random_state: None,
            layers: Vec::new(),
            classifier_weights: Array2::zeros((0, 0)),
            classifier_bias: Array1::zeros(0),
            n_classes: 0,
            is_pretrained: false,
            is_finetuned: false,
        }
    }
}

impl StackedAutoencoders {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn layer_sizes(mut self, layer_sizes: Vec<usize>) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(StackedAutoencoderError::InvalidLayerSizes(layer_sizes).into());
        }
        self.layer_sizes = layer_sizes;
        Ok(self)
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(StackedAutoencoderError::InvalidLearningRate(learning_rate).into());
        }
        self.learning_rate = learning_rate;
        Ok(self)
    }

    pub fn pretrain_epochs(mut self, pretrain_epochs: usize) -> Result<Self> {
        if pretrain_epochs == 0 {
            return Err(StackedAutoencoderError::InvalidEpochs(pretrain_epochs).into());
        }
        self.pretrain_epochs = pretrain_epochs;
        Ok(self)
    }

    pub fn finetune_epochs(mut self, finetune_epochs: usize) -> Result<Self> {
        if finetune_epochs == 0 {
            return Err(StackedAutoencoderError::InvalidEpochs(finetune_epochs).into());
        }
        self.finetune_epochs = finetune_epochs;
        Ok(self)
    }

    pub fn noise_factor(mut self, noise_factor: f64) -> Self {
        self.noise_factor = noise_factor;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    fn initialize_layers(&mut self) {
        self.layers.clear();

        for i in 0..self.layer_sizes.len() - 1 {
            let layer = AutoencoderLayer::new(self.layer_sizes[i], self.layer_sizes[i + 1])
                .learning_rate(self.learning_rate)
                .unwrap()
                .epochs(self.pretrain_epochs)
                .unwrap()
                .noise_factor(self.noise_factor);
            self.layers.push(layer);
        }
    }

    pub fn pretrain(&mut self, X: &ArrayView2<f64>) -> Result<()> {
        let (_, n_features) = X.dim();

        // Set input size from data
        if self.layer_sizes.is_empty() {
            self.layer_sizes = vec![n_features, n_features / 2, n_features / 4];
        } else {
            self.layer_sizes[0] = n_features;
        }

        self.initialize_layers();

        // Layer-wise pre-training
        let mut current_data = X.to_owned();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let seed = self.random_state.map(|s| s + i as u64);
            layer.fit(&current_data.view(), seed)?;

            // Transform data for next layer
            current_data = layer.transform(&current_data.view())?;
        }

        self.is_pretrained = true;
        Ok(())
    }

    fn forward_pass(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if !self.is_pretrained {
            return Err(StackedAutoencoderError::ModelNotTrained.into());
        }

        let mut current_data = X.to_owned();

        for layer in self.layers.iter() {
            current_data = layer.transform(&current_data.view())?;
        }

        Ok(current_data)
    }

    fn softmax(&self, X: &ArrayView2<f64>) -> Array2<f64> {
        let mut result = X.to_owned();

        for mut row in result.rows_mut() {
            let max_val = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
            }
            let sum: f64 = row.sum();
            for val in row.iter_mut() {
                *val /= sum;
            }
        }

        result
    }

    pub fn finetune(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<()> {
        if !self.is_pretrained {
            return Err(StackedAutoencoderError::ModelNotTrained.into());
        }

        let (n_samples, _) = X.dim();

        if y.len() != n_samples {
            return Err(StackedAutoencoderError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
            }
            .into());
        }

        // Count labeled samples
        let labeled_mask: Vec<bool> = y.iter().map(|&label| label >= 0).collect();
        let n_labeled = labeled_mask.iter().filter(|&&x| x).count();

        if n_labeled == 0 {
            return Err(StackedAutoencoderError::InsufficientLabeledSamples.into());
        }

        // Get unique classes from labeled data
        let mut classes: Vec<i32> = y.iter().filter(|&&label| label >= 0).cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        self.n_classes = classes.len();

        // Initialize classifier weights
        let hidden_size = self.layer_sizes.last().unwrap();
        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let limit = (6.0 / (hidden_size + self.n_classes) as f64).sqrt();

        // Initialize classifier weights manually
        let mut classifier_weights = Array2::<f64>::zeros((*hidden_size, self.n_classes));
        for i in 0..*hidden_size {
            for j in 0..self.n_classes {
                // Generate uniform distributed random number in [-limit, limit]
                let u: f64 = rng.random_range(0.0, 1.0);
                classifier_weights[(i, j)] = u * (2.0 * limit) - limit;
            }
        }
        self.classifier_weights = classifier_weights;
        self.classifier_bias = Array1::zeros(self.n_classes);

        // Fine-tuning with labeled data
        for _epoch in 0..self.finetune_epochs {
            // Forward pass through pre-trained layers
            let features = self.forward_pass(X)?;

            // Classifier forward pass
            let logits = features.dot(&self.classifier_weights) + &self.classifier_bias;
            let probabilities = self.softmax(&logits.view());

            // Compute loss and gradients only for labeled samples
            let mut classifier_weights_grad = Array2::zeros(self.classifier_weights.raw_dim());
            let mut classifier_bias_grad = Array1::zeros(self.classifier_bias.len());

            for i in 0..n_samples {
                if labeled_mask[i] {
                    let true_class = y[i] as usize;

                    // Cross-entropy gradient
                    for j in 0..self.n_classes {
                        let target = if j == true_class { 1.0 } else { 0.0 };
                        let error = probabilities[[i, j]] - target;

                        // Update gradients
                        for k in 0..*hidden_size {
                            classifier_weights_grad[[k, j]] += features[[i, k]] * error;
                        }
                        classifier_bias_grad[j] += error;
                    }
                }
            }

            // Update classifier weights
            classifier_weights_grad /= n_labeled as f64;
            classifier_bias_grad /= n_labeled as f64;

            self.classifier_weights =
                &self.classifier_weights - self.learning_rate * classifier_weights_grad;
            self.classifier_bias =
                &self.classifier_bias - self.learning_rate * classifier_bias_grad;
        }

        self.is_finetuned = true;
        Ok(())
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>> {
        self.forward_pass(X)
    }

    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if !self.is_finetuned {
            return Err(StackedAutoencoderError::ModelNotTrained.into());
        }

        let features = self.forward_pass(X)?;
        let logits = features.dot(&self.classifier_weights) + &self.classifier_bias;
        Ok(self.softmax(&logits.view()))
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let predictions = probabilities.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i32)
                .unwrap()
        });
        Ok(predictions)
    }
}

#[derive(Debug, Clone)]
pub struct FittedStackedAutoencoders {
    model: StackedAutoencoders,
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>, FittedStackedAutoencoders>
    for StackedAutoencoders
{
    type Fitted = FittedStackedAutoencoders;

    fn fit(
        mut self,
        X: &ArrayView2<'_, f64>,
        y: &ArrayView1<'_, i32>,
    ) -> Result<FittedStackedAutoencoders> {
        // Pre-train with all data (including unlabeled)
        self.pretrain(X)?;

        // Fine-tune with labeled data
        self.finetune(X, y)?;

        Ok(FittedStackedAutoencoders { model: self })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for FittedStackedAutoencoders {
    fn predict(&self, X: &ArrayView2<'_, f64>) -> Result<Array1<i32>> {
        self.model.predict(X)
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for FittedStackedAutoencoders {
    fn transform(&self, X: &ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.model.transform(X)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    fn test_autoencoder_layer_creation() {
        let layer = AutoencoderLayer::new(10, 5);
        assert_eq!(layer.input_size, 10);
        assert_eq!(layer.hidden_size, 5);
        assert_eq!(layer.learning_rate, 0.001);
        assert_eq!(layer.epochs, 100);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_autoencoder_layer_fit_transform() {
        let mut layer = AutoencoderLayer::new(4, 2)
            .learning_rate(0.01)
            .unwrap()
            .epochs(50)
            .unwrap();

        let X = array![
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ];

        layer.fit(&X.view(), Some(42)).unwrap();
        assert!(layer.is_trained);

        let encoded = layer.transform(&X.view()).unwrap();
        assert_eq!(encoded.dim(), (4, 2));

        let reconstructed = layer.reconstruct(&X.view()).unwrap();
        assert_eq!(reconstructed.dim(), (4, 4));
    }

    #[test]
    fn test_stacked_autoencoders_creation() {
        let sae = StackedAutoencoders::new()
            .layer_sizes(vec![4, 3, 2])
            .unwrap()
            .learning_rate(0.01)
            .unwrap()
            .pretrain_epochs(10)
            .unwrap()
            .finetune_epochs(5)
            .unwrap();

        assert_eq!(sae.layer_sizes, vec![4, 3, 2]);
        assert_eq!(sae.learning_rate, 0.01);
        assert_eq!(sae.pretrain_epochs, 10);
        assert_eq!(sae.finetune_epochs, 5);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stacked_autoencoders_pretrain() {
        let mut sae = StackedAutoencoders::new()
            .layer_sizes(vec![4, 3, 2])
            .unwrap()
            .pretrain_epochs(5)
            .unwrap()
            .random_state(42);

        let X = array![
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ];

        sae.pretrain(&X.view()).unwrap();
        assert!(sae.is_pretrained);
        assert_eq!(sae.layers.len(), 2); // 4->3 and 3->2

        let features = sae.transform(&X.view()).unwrap();
        assert_eq!(features.dim(), (4, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stacked_autoencoders_fit_predict() {
        let sae = StackedAutoencoders::new()
            .layer_sizes(vec![4, 3, 2])
            .unwrap()
            .pretrain_epochs(5)
            .unwrap()
            .finetune_epochs(3)
            .unwrap()
            .random_state(42);

        let X = array![
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.2, 0.8, 0.3, 0.7]
        ];
        let y = array![0, 1, 0, 1, -1, -1]; // -1 indicates unlabeled

        let fitted = sae.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 6);

        let probabilities = fitted.model.predict_proba(&X.view()).unwrap();
        assert_eq!(probabilities.dim(), (6, 2));

        // Check that probabilities sum to 1
        for i in 0..6 {
            let sum: f64 = probabilities.row(i).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_stacked_autoencoders_invalid_parameters() {
        assert!(StackedAutoencoders::new().layer_sizes(vec![]).is_err());
        assert!(StackedAutoencoders::new().layer_sizes(vec![10]).is_err());
        assert!(StackedAutoencoders::new().learning_rate(0.0).is_err());
        assert!(StackedAutoencoders::new().learning_rate(-0.1).is_err());
        assert!(StackedAutoencoders::new().pretrain_epochs(0).is_err());
        assert!(StackedAutoencoders::new().finetune_epochs(0).is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stacked_autoencoders_insufficient_labeled_samples() {
        let sae = StackedAutoencoders::new()
            .layer_sizes(vec![4, 2])
            .unwrap()
            .pretrain_epochs(2)
            .unwrap()
            .finetune_epochs(2)
            .unwrap();

        let X = array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]];
        let y = array![-1, -1]; // All unlabeled

        let result = sae.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stacked_autoencoders_transform() {
        let sae = StackedAutoencoders::new()
            .layer_sizes(vec![4, 2])
            .unwrap()
            .pretrain_epochs(3)
            .unwrap()
            .finetune_epochs(2)
            .unwrap()
            .random_state(42);

        let X = array![
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0]
        ];
        let y = array![0, 1, 0];

        let fitted = sae.fit(&X.view(), &y.view()).unwrap();
        let features = fitted.transform(&X.view()).unwrap();

        assert_eq!(features.dim(), (3, 2));

        // Check that features are within reasonable bounds
        for value in features.iter() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
    }
}
