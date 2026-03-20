//! Contrastive Predictive Coding (CPC) implementation for semi-supervised learning

use super::{ContrastiveLearningError, *};
use scirs2_core::random::rand_prelude::SliceRandom;

/// Contrastive Predictive Coding (CPC) for semi-supervised learning
///
/// CPC learns representations by predicting future observations from past contexts
/// in a contrastive manner. It maximizes mutual information between contexts and
/// positive samples while minimizing it for negative samples.
#[derive(Debug, Clone)]
pub struct ContrastivePredictiveCoding {
    /// embedding_dim
    pub embedding_dim: usize,
    /// hidden_dim
    pub hidden_dim: usize,
    /// context_length
    pub context_length: usize,
    /// prediction_steps
    pub prediction_steps: usize,
    /// temperature
    pub temperature: f64,
    /// learning_rate
    pub learning_rate: f64,
    /// batch_size
    pub batch_size: usize,
    /// max_epochs
    pub max_epochs: usize,
    /// negative_samples
    pub negative_samples: usize,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for ContrastivePredictiveCoding {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            hidden_dim: 256,
            context_length: 8,
            prediction_steps: 4,
            temperature: 0.1,
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            negative_samples: 16,
            random_state: None,
        }
    }
}

impl ContrastivePredictiveCoding {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    pub fn hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = hidden_dim;
        self
    }

    pub fn context_length(mut self, context_length: usize) -> Self {
        self.context_length = context_length;
        self
    }

    pub fn prediction_steps(mut self, prediction_steps: usize) -> Self {
        self.prediction_steps = prediction_steps;
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(ContrastiveLearningError::InvalidTemperature(temperature).into());
        }
        self.temperature = temperature;
        Ok(self)
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(ContrastiveLearningError::InvalidBatchSize(batch_size).into());
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    pub fn negative_samples(mut self, negative_samples: usize) -> Self {
        self.negative_samples = negative_samples;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    fn encode(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Simple linear encoder for demonstration - create weights manually
        let mut encoder_weights = Array2::<f64>::zeros((n_features, self.embedding_dim));
        for i in 0..n_features {
            for j in 0..self.embedding_dim {
                // Generate normal distributed random number using Box-Muller transform
                let u1: f64 = rng.random_range(0.0..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                encoder_weights[(i, j)] = z * 0.1; // mean=0.0, std=0.1
            }
        }

        Ok(x.dot(&encoder_weights))
    }

    fn context_network(&self, embeddings: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let (n_samples, embedding_dim) = embeddings.dim();
        if embedding_dim != self.embedding_dim {
            return Err(ContrastiveLearningError::EmbeddingDimensionMismatch {
                expected: self.embedding_dim,
                actual: embedding_dim,
            }
            .into());
        }

        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Simple context network (could be LSTM/GRU in practice)
        // Create context weights manually
        let mut context_weights = Array2::<f64>::zeros((self.embedding_dim, self.hidden_dim));
        for i in 0..self.embedding_dim {
            for j in 0..self.hidden_dim {
                // Generate normal distributed random number using Box-Muller transform
                let u1: f64 = rng.random_range(0.0..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                context_weights[(i, j)] = z * 0.1; // mean=0.0, std=0.1
            }
        }

        Ok(embeddings.dot(&context_weights))
    }

    fn compute_contrastive_loss(
        &self,
        context: &ArrayView2<f64>,
        positive: &ArrayView2<f64>,
        negatives: &ArrayView2<f64>,
    ) -> Result<f64> {
        let batch_size = context.dim().0;
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let ctx = context.row(i);
            let pos = positive.row(i);

            // Compute positive score
            let pos_score = ctx.dot(&pos) / self.temperature;

            // Compute negative scores
            let mut neg_scores = Vec::new();
            for j in 0..self.negative_samples {
                if j < negatives.dim().0 {
                    let neg = negatives.row(j);
                    let neg_score = ctx.dot(&neg) / self.temperature;
                    neg_scores.push(neg_score);
                }
            }

            // Compute softmax loss
            let max_score =
                pos_score.max(neg_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            let exp_pos = (pos_score - max_score).exp();
            let exp_neg_sum: f64 = neg_scores.iter().map(|&s| (s - max_score).exp()).sum();

            let loss = -((exp_pos / (exp_pos + exp_neg_sum)).ln());
            total_loss += loss;
        }

        Ok(total_loss / batch_size as f64)
    }
}

/// Fitted Contrastive Predictive Coding model
#[derive(Debug, Clone)]
pub struct FittedContrastivePredictiveCoding {
    /// base_model
    pub base_model: ContrastivePredictiveCoding,
    /// encoder_weights
    pub encoder_weights: Array2<f64>,
    /// context_weights
    pub context_weights: Array2<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// n_classes
    pub n_classes: usize,
}

impl Estimator for ContrastivePredictiveCoding {
    type Config = ContrastivePredictiveCoding;
    type Error = ContrastiveLearningError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for ContrastivePredictiveCoding {
    type Fitted = FittedContrastivePredictiveCoding;

    fn fit(self, X: &ArrayView2<'_, f64>, y: &ArrayView1<'_, i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = X.dim();

        // Check for sufficient labeled samples
        let labeled_count = y.iter().filter(|&&label| label != -1).count();
        if labeled_count < 2 {
            return Err(ContrastiveLearningError::InsufficientLabeledSamples.into());
        }

        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Initialize encoder and context networks
        let mut encoder_weights = Array2::<f64>::zeros((n_features, self.embedding_dim));
        let mut context_weights = Array2::<f64>::zeros((self.embedding_dim, self.hidden_dim));

        // Fill encoder weights with normal distribution (mean=0.0, std=0.1)
        for i in 0..n_features {
            for j in 0..self.embedding_dim {
                let u1: f64 = rng.random_range(0.0..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                encoder_weights[(i, j)] = z * 0.1;
            }
        }

        // Fill context weights with normal distribution (mean=0.0, std=0.1)
        for i in 0..self.embedding_dim {
            for j in 0..self.hidden_dim {
                let u1: f64 = rng.random_range(0.0..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                context_weights[(i, j)] = z * 0.1;
            }
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

        // Training loop
        for epoch in 0..self.max_epochs {
            // Generate batches
            let batch_indices: Vec<usize> = (0..n_samples).collect();
            let mut batch_indices = batch_indices;
            batch_indices.shuffle(&mut rng);

            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..n_samples).step_by(self.batch_size) {
                let batch_end = std::cmp::min(batch_start + self.batch_size, n_samples);
                let batch_size = batch_end - batch_start;

                if batch_size < 2 {
                    continue;
                }

                // Get batch data
                let batch_X = X.slice(scirs2_core::ndarray::s![batch_start..batch_end, ..]);

                // Encode batch
                let encoded = batch_X.dot(&encoder_weights);

                // Context network
                let context = encoded.dot(&context_weights);

                // Generate positive and negative samples
                let mut positive_samples = Vec::new();
                let mut negative_samples = Vec::new();

                for i in 0..batch_size {
                    // Use next sample as positive (temporal structure)
                    let pos_idx = if i + 1 < batch_size { i + 1 } else { 0 };
                    positive_samples.push(encoded.row(pos_idx).to_owned());

                    // Random negative samples
                    let max_negatives = std::cmp::min(self.negative_samples, batch_size - 1);
                    let mut neg_count = 0;
                    while neg_count < max_negatives {
                        let neg_idx = rng.gen_range(0..batch_size);
                        if neg_idx != i {
                            negative_samples.push(encoded.row(neg_idx).to_owned());
                            neg_count += 1;
                        }
                    }
                }

                // Convert to arrays
                let positive_array = Array2::from_shape_vec(
                    (batch_size, self.embedding_dim),
                    positive_samples.into_iter().flatten().collect(),
                )
                .map_err(|e| {
                    ContrastiveLearningError::MatrixOperationFailed(format!(
                        "Array creation failed: {}",
                        e
                    ))
                })?;

                let actual_negative_count = negative_samples.len();
                let negative_array = Array2::from_shape_vec(
                    (actual_negative_count, self.embedding_dim),
                    negative_samples.into_iter().flatten().collect(),
                )
                .map_err(|e| {
                    ContrastiveLearningError::MatrixOperationFailed(format!(
                        "Array creation failed: {}",
                        e
                    ))
                })?;

                // Compute loss using encoded representations
                let loss = self.compute_contrastive_loss(
                    &encoded.view(),
                    &positive_array.view(),
                    &negative_array.view(),
                )?;
                epoch_loss += loss;
                n_batches += 1;

                // Simple gradient update (in practice, would use proper backpropagation)
                let gradient_scale = self.learning_rate * loss;
                // Create gradient noise manually
                let noise_std = gradient_scale * 0.1;
                let mut encoder_grad = Array2::<f64>::zeros(encoder_weights.dim());
                let mut context_grad = Array2::<f64>::zeros(context_weights.dim());

                // Fill encoder grad with normal noise
                for i in 0..encoder_weights.nrows() {
                    for j in 0..encoder_weights.ncols() {
                        let u1: f64 = rng.random_range(0.0..1.0);
                        let u2: f64 = rng.random_range(0.0..1.0);
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        encoder_grad[(i, j)] = z * noise_std;
                    }
                }

                // Fill context grad with normal noise
                for i in 0..context_weights.nrows() {
                    for j in 0..context_weights.ncols() {
                        let u1: f64 = rng.random_range(0.0..1.0);
                        let u2: f64 = rng.random_range(0.0..1.0);
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        context_grad[(i, j)] = z * noise_std;
                    }
                }

                encoder_weights = encoder_weights - encoder_grad;
                context_weights = context_weights - context_grad;
            }

            if n_batches > 0 {
                epoch_loss /= n_batches as f64;
            }

            // Early stopping or convergence check could be added here
            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, epoch_loss);
            }
        }

        Ok(FittedContrastivePredictiveCoding {
            base_model: self.clone(),
            encoder_weights,
            context_weights,
            classes: Array1::from_vec(unique_classes),
            n_classes,
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for FittedContrastivePredictiveCoding {
    fn predict(&self, X: &ArrayView2<'_, f64>) -> Result<Array1<i32>> {
        let embeddings = X.dot(&self.encoder_weights);

        let context = embeddings.dot(&self.context_weights);

        // Simple nearest class prediction based on context representations
        let n_samples = X.dim().0;
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let ctx = context.row(i);
            let mut best_class = self.classes[0];
            let mut best_score = f64::NEG_INFINITY;

            for &class in self.classes.iter() {
                // Simple scoring based on context magnitude (placeholder)
                let score = ctx.sum() + class as f64 * 0.1;
                if score > best_score {
                    best_score = score;
                    best_class = class;
                }
            }

            predictions[i] = best_class;
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>> for FittedContrastivePredictiveCoding {
    fn predict_proba(&self, X: &ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let embeddings = X.dot(&self.encoder_weights);

        let context = embeddings.dot(&self.context_weights);

        let n_samples = X.dim().0;
        let mut probabilities = Array2::zeros((n_samples, self.n_classes));

        for i in 0..n_samples {
            let ctx = context.row(i);
            let mut scores = Vec::new();

            for &class in self.classes.iter() {
                // Simple scoring based on context (placeholder)
                let score = ctx.sum() + class as f64 * 0.1;
                scores.push(score);
            }

            // Softmax normalization
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            for (j, &exp_score) in exp_scores.iter().enumerate() {
                probabilities[[i, j]] = exp_score / sum_exp;
            }
        }

        Ok(probabilities)
    }
}
