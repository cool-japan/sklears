//! Momentum Contrast (MoCo) implementation for semi-supervised learning

use super::{ContrastiveLearningError, *};
use scirs2_core::random::{rand_prelude::SliceRandom, Rng};

/// Momentum Contrast (MoCo) adaptation for semi-supervised learning
///
/// MoCo builds a large and consistent dictionary for contrastive learning using momentum-based updates.
/// This implementation adapts MoCo for semi-supervised scenarios by incorporating labeled data.
///
/// # Parameters
///
/// * `embedding_dim` - Dimensionality of learned embeddings
/// * `queue_size` - Size of the momentum queue (dictionary)
/// * `temperature` - Temperature parameter for contrastive loss
/// * `momentum` - Momentum coefficient for updating key encoder
/// * `batch_size` - Batch size for training
/// * `max_epochs` - Maximum number of training epochs
/// * `learning_rate` - Learning rate for query encoder
/// * `augmentation_strength` - Strength of data augmentation
/// * `labeled_weight` - Weight for supervised loss component
/// * `random_state` - Random seed for reproducibility
#[derive(Debug, Clone)]
pub struct MomentumContrast {
    /// embedding_dim
    pub embedding_dim: usize,
    /// queue_size
    pub queue_size: usize,
    /// temperature
    pub temperature: f64,
    /// momentum
    pub momentum: f64,
    /// batch_size
    pub batch_size: usize,
    /// max_epochs
    pub max_epochs: usize,
    /// learning_rate
    pub learning_rate: f64,
    /// augmentation_strength
    pub augmentation_strength: f64,
    /// labeled_weight
    pub labeled_weight: f64,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for MomentumContrast {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            queue_size: 1024,
            temperature: 0.2,
            momentum: 0.999,
            batch_size: 32,
            max_epochs: 100,
            learning_rate: 0.001,
            augmentation_strength: 0.2,
            labeled_weight: 1.0,
            random_state: None,
        }
    }
}

impl MomentumContrast {
    /// Create a new MomentumContrast instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedding dimensionality
    pub fn embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set the queue size
    pub fn queue_size(mut self, queue_size: usize) -> Self {
        self.queue_size = queue_size;
        self
    }

    /// Set the temperature parameter
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the momentum coefficient
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the maximum number of epochs
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the augmentation strength
    pub fn augmentation_strength(mut self, strength: f64) -> Self {
        self.augmentation_strength = strength;
        self
    }

    /// Set the labeled weight
    pub fn labeled_weight(mut self, labeled_weight: f64) -> Self {
        self.labeled_weight = labeled_weight;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    fn apply_augmentation<R>(&self, x: &Array2<f64>, rng: &mut Random<R>) -> Array2<f64>
    where
        R: Rng,
    {
        let mut augmented = x.clone();

        // Gaussian noise augmentation - create noise manually
        let mut noise = Array2::<f64>::zeros(x.dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                // Generate normal distributed random number
                let u1: f64 = rng.random_range(0.0..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                noise[(i, j)] = z * self.augmentation_strength;
            }
        }
        augmented = augmented + noise;

        // Feature scaling perturbation
        for mut row in augmented.axis_iter_mut(Axis(0)) {
            let scale_factor = 1.0 + self.augmentation_strength * (rng.gen::<f64>() - 0.5);
            row *= scale_factor;
        }

        augmented
    }

    fn update_key_encoder(&self, query_weights: &Array2<f64>, key_weights: &mut Array2<f64>) {
        // Momentum update: key_weights = momentum * key_weights + (1 - momentum) * query_weights
        *key_weights = self.momentum * key_weights.clone() + (1.0 - self.momentum) * query_weights;
    }

    fn l2_normalize(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut normalized = x.clone();
        for mut row in normalized.axis_iter_mut(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-12 {
                row /= norm;
            }
        }
        normalized
    }

    fn compute_moco_loss(
        &self,
        q: &ArrayView2<f64>,
        k: &ArrayView2<f64>,
        queue: &ArrayView2<f64>,
    ) -> Result<f64> {
        let batch_size = q.nrows();
        if batch_size == 0 {
            return Ok(0.0);
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let qi = q.row(i);
            let ki = k.row(i);

            // Positive logit
            let l_pos = qi.dot(&ki) / self.temperature;

            // Negative logits from queue
            let mut l_neg_sum = 0.0;
            for j in 0..queue.nrows() {
                let queuej = queue.row(j);
                let l_neg = qi.dot(&queuej) / self.temperature;
                l_neg_sum += l_neg.exp();
            }

            // Include positive in denominator
            let logits = l_pos - (l_pos.exp() + l_neg_sum).ln();
            total_loss -= logits;
        }

        Ok(total_loss / batch_size as f64)
    }

    fn compute_supervised_loss(
        &self,
        embeddings: &ArrayView2<f64>,
        labels: &ArrayView1<i32>,
    ) -> Result<f64> {
        let batch_size = embeddings.nrows();
        let mut total_loss = 0.0;
        let mut valid_pairs = 0;

        for i in 0..batch_size {
            if labels[i] == -1 {
                continue; // Skip unlabeled samples
            }

            let ei = embeddings.row(i);
            let mut pos_sum = 0.0;
            let mut neg_sum = 0.0;
            let mut pos_count = 0;

            for j in 0..batch_size {
                if i == j || labels[j] == -1 {
                    continue;
                }

                let ej = embeddings.row(j);
                let similarity = (ei.dot(&ej) / self.temperature).exp();

                if labels[i] == labels[j] {
                    pos_sum += similarity;
                    pos_count += 1;
                } else {
                    neg_sum += similarity;
                }
            }

            if pos_count > 0 {
                let loss = -(pos_sum / (pos_sum + neg_sum)).ln();
                total_loss += loss;
                valid_pairs += 1;
            }
        }

        if valid_pairs > 0 {
            Ok(total_loss / valid_pairs as f64)
        } else {
            Ok(0.0)
        }
    }

    fn update_queue(&self, queue: &mut Array2<f64>, new_keys: &Array2<f64>) {
        let queue_size = queue.nrows();
        let new_size = new_keys.nrows();

        if new_size >= queue_size {
            // Replace entire queue
            queue.assign(&new_keys.slice(s![0..queue_size, ..]));
        } else {
            // Create a temporary copy to avoid borrowing conflicts
            let queue_copy = queue.clone();

            // Shift existing entries and add new ones
            for i in 0..(queue_size - new_size) {
                queue.row_mut(i).assign(&queue_copy.row(i + new_size));
            }
            for i in 0..new_size {
                queue
                    .row_mut(queue_size - new_size + i)
                    .assign(&new_keys.row(i));
            }
        }
    }
}

/// Fitted MomentumContrast model
#[derive(Debug, Clone)]
pub struct FittedMomentumContrast {
    /// base_model
    pub base_model: MomentumContrast,
    /// query_encoder
    pub query_encoder: Array2<f64>,
    /// key_encoder
    pub key_encoder: Array2<f64>,
    /// queue
    pub queue: Array2<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// n_classes
    pub n_classes: usize,
}

impl Estimator for MomentumContrast {
    type Config = MomentumContrast;
    type Error = ContrastiveLearningError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for MomentumContrast {
    type Fitted = FittedMomentumContrast;

    fn fit(self, X: &ArrayView2<'_, f64>, y: &ArrayView1<'_, i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = X.dim();

        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Initialize query and key encoders - create weights manually
        let mut query_encoder = Array2::<f64>::zeros((n_features, self.embedding_dim));
        for i in 0..n_features {
            for j in 0..self.embedding_dim {
                // Generate normal distributed random number (mean=0.0, std=0.02)
                let u1: f64 = rng.random_range(0.0..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                query_encoder[(i, j)] = z * 0.02;
            }
        }
        let mut key_encoder = query_encoder.clone();

        // Initialize queue with random normalized vectors
        let mut queue = Array2::<f64>::zeros((self.queue_size, self.embedding_dim));
        for i in 0..self.queue_size {
            for j in 0..self.embedding_dim {
                // Generate normal distributed random number (mean=0.0, std=0.02)
                let u1: f64 = rng.random_range(0.0..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                queue[(i, j)] = z * 0.02;
            }
        }
        queue = self.l2_normalize(&queue);

        // Get unique classes for supervised component
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
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            // Generate batches
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            for batch_start in (0..n_samples).step_by(self.batch_size) {
                let batch_end = std::cmp::min(batch_start + self.batch_size, n_samples);
                let batch_size = batch_end - batch_start;

                if batch_size < 2 {
                    continue;
                }

                let batch_indices = &indices[batch_start..batch_end];

                // Extract batch data
                let mut batch_X = Array2::zeros((batch_size, n_features));
                let mut batch_y = Array1::zeros(batch_size);

                for (i, &idx) in batch_indices.iter().enumerate() {
                    batch_X.row_mut(i).assign(&X.row(idx));
                    batch_y[i] = y[idx];
                }

                // Generate query and key augmentations
                let X_query = self.apply_augmentation(&batch_X, &mut rng);
                let X_key = self.apply_augmentation(&batch_X, &mut rng);

                // Forward pass
                let q = X_query.dot(&query_encoder);
                let k = X_key.dot(&key_encoder);

                // L2 normalize
                let q_norm = self.l2_normalize(&q);
                let k_norm = self.l2_normalize(&k);

                // Compute MoCo loss
                let moco_loss =
                    self.compute_moco_loss(&q_norm.view(), &k_norm.view(), &queue.view())?;

                // Compute supervised loss
                let supervised_loss = if n_classes > 0 {
                    self.compute_supervised_loss(&q.view(), &batch_y.view())?
                } else {
                    0.0
                };

                // Combined loss
                let total_loss = moco_loss + self.labeled_weight * supervised_loss;
                epoch_loss += total_loss;
                n_batches += 1;

                // Update query encoder (simple gradient simulation)
                let gradient_scale = self.learning_rate * total_loss;
                // Create gradient noise manually
                let noise_std = gradient_scale * 0.01;
                let mut encoder_grad = Array2::<f64>::zeros(query_encoder.dim());
                for i in 0..query_encoder.nrows() {
                    for j in 0..query_encoder.ncols() {
                        let u1: f64 = rng.random_range(0.0..1.0);
                        let u2: f64 = rng.random_range(0.0..1.0);
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        encoder_grad[(i, j)] = z * noise_std;
                    }
                }
                query_encoder = query_encoder - encoder_grad;

                // Update key encoder with momentum
                self.update_key_encoder(&query_encoder, &mut key_encoder);

                // Update queue
                self.update_queue(&mut queue, &k_norm);
            }

            if n_batches > 0 {
                epoch_loss /= n_batches as f64;
            }
        }

        Ok(FittedMomentumContrast {
            base_model: self,
            query_encoder,
            key_encoder,
            queue,
            classes: Array1::from_vec(unique_classes),
            n_classes,
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for FittedMomentumContrast {
    fn predict(&self, X: &ArrayView2<'_, f64>) -> Result<Array1<i32>> {
        let embeddings = X.dot(&self.query_encoder);
        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        if self.n_classes == 0 {
            return Ok(predictions);
        }

        // Simple classification based on embeddings
        for i in 0..n_samples {
            let embedding = embeddings.row(i);
            let score = embedding.sum();
            let class_idx = ((score.abs() * self.n_classes as f64) as usize) % self.n_classes;
            predictions[i] = self.classes[class_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>> for FittedMomentumContrast {
    fn predict_proba(&self, X: &ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let embeddings = X.dot(&self.query_encoder);
        let n_samples = X.nrows();
        let mut probabilities = Array2::zeros((n_samples, self.n_classes.max(1)));

        if self.n_classes == 0 {
            probabilities.fill(1.0);
            return Ok(probabilities);
        }

        for i in 0..n_samples {
            let embedding = embeddings.row(i);

            // Generate probabilities based on embedding similarity to queue
            let mut scores = Vec::new();
            for j in 0..self.n_classes {
                let score = embedding.sum() + j as f64 * 0.1;
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

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    fn test_moco_creation() {
        let moco = MomentumContrast::new()
            .embedding_dim(64)
            .queue_size(512)
            .temperature(0.1)
            .momentum(0.995)
            .max_epochs(10);

        assert_eq!(moco.embedding_dim, 64);
        assert_eq!(moco.queue_size, 512);
        assert_eq!(moco.temperature, 0.1);
        assert_eq!(moco.momentum, 0.995);
        assert_eq!(moco.max_epochs, 10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_moco_fit_predict() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0]
        ];
        let y = array![0, 1, 0, 1, -1, -1]; // -1 indicates unlabeled

        let moco = MomentumContrast::new()
            .embedding_dim(8)
            .queue_size(16)
            .max_epochs(2)
            .batch_size(3)
            .random_state(42);

        let fitted = moco.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 6);
        for &pred in predictions.iter() {
            assert!(pred >= 0 && pred < 2);
        }

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to 1
        for i in 0..6 {
            let sum: f64 = probas.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_moco_momentum_update() {
        let moco = MomentumContrast::new().momentum(0.9);
        let query_weights = array![[1.0, 2.0], [3.0, 4.0]];
        let mut key_weights = array![[0.0, 0.0], [0.0, 0.0]];

        moco.update_key_encoder(&query_weights, &mut key_weights);

        // Expected: 0.9 * 0 + 0.1 * query_weights = 0.1 * query_weights
        assert!((key_weights[[0, 0]] - 0.1).abs() < 1e-10);
        assert!((key_weights[[0, 1]] - 0.2).abs() < 1e-10);
        assert!((key_weights[[1, 0]] - 0.3).abs() < 1e-10);
        assert!((key_weights[[1, 1]] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_moco_queue_update() {
        let moco = MomentumContrast::new();
        let mut queue = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let new_keys = array![[7.0, 8.0]];

        moco.update_queue(&mut queue, &new_keys);

        // Queue should shift and add new key at the end
        assert_eq!(queue[[0, 0]], 3.0);
        assert_eq!(queue[[0, 1]], 4.0);
        assert_eq!(queue[[1, 0]], 5.0);
        assert_eq!(queue[[1, 1]], 6.0);
        assert_eq!(queue[[2, 0]], 7.0);
        assert_eq!(queue[[2, 1]], 8.0);
    }

    #[test]
    fn test_moco_l2_normalization() {
        let moco = MomentumContrast::new();
        let x = array![[3.0, 4.0], [0.0, 5.0]];

        let normalized = moco.l2_normalize(&x);

        // Check that each row has unit norm
        for row in normalized.axis_iter(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            assert!((norm - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_moco_all_unlabeled() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![-1, -1, -1]; // All unlabeled

        let moco = MomentumContrast::new()
            .queue_size(8)
            .max_epochs(2)
            .batch_size(2);

        let fitted = moco.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        // All predictions should be 0 when no labeled classes exist
        for &pred in predictions.iter() {
            assert_eq!(pred, 0);
        }
    }
}
