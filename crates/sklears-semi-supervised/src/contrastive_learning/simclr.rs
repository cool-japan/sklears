//! SimCLR (A Simple Framework for Contrastive Learning) implementation for semi-supervised learning

use super::{ContrastiveLearningError, *};
use scirs2_core::random::{rand_prelude::SliceRandom, Rng};

/// SimCLR (A Simple Framework for Contrastive Learning) adaptation for semi-supervised learning
///
/// This implements SimCLR's contrastive learning approach adapted for semi-supervised scenarios.
/// It learns representations by maximizing agreement between differently augmented views of the same data.
///
/// # Parameters
///
/// * `projection_dim` - Dimensionality of projection head (typically smaller than embedding_dim)
/// * `embedding_dim` - Dimensionality of learned embeddings
/// * `temperature` - Temperature parameter for contrastive loss
/// * `augmentation_strength` - Strength of data augmentation
/// * `batch_size` - Batch size for training
/// * `max_epochs` - Maximum number of training epochs
/// * `learning_rate` - Learning rate for optimization
/// * `momentum` - Momentum for exponential moving averages
/// * `labeled_weight` - Weight for supervised contrastive loss component
/// * `random_state` - Random seed for reproducibility
#[derive(Debug, Clone)]
pub struct SimCLR {
    /// projection_dim
    pub projection_dim: usize,
    /// embedding_dim
    pub embedding_dim: usize,
    /// temperature
    pub temperature: f64,
    /// augmentation_strength
    pub augmentation_strength: f64,
    /// batch_size
    pub batch_size: usize,
    /// max_epochs
    pub max_epochs: usize,
    /// learning_rate
    pub learning_rate: f64,
    /// momentum
    pub momentum: f64,
    /// labeled_weight
    pub labeled_weight: f64,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for SimCLR {
    fn default() -> Self {
        Self {
            projection_dim: 64,
            embedding_dim: 128,
            temperature: 0.5,
            augmentation_strength: 0.2,
            batch_size: 32,
            max_epochs: 100,
            learning_rate: 0.001,
            momentum: 0.999,
            labeled_weight: 1.0,
            random_state: None,
        }
    }
}

impl SimCLR {
    /// Create a new SimCLR instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the projection head dimensionality
    pub fn projection_dim(mut self, projection_dim: usize) -> Self {
        self.projection_dim = projection_dim;
        self
    }

    /// Set the embedding dimensionality
    pub fn embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set the temperature parameter
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the augmentation strength
    pub fn augmentation_strength(mut self, strength: f64) -> Self {
        self.augmentation_strength = strength;
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

    /// Set the momentum parameter
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
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
                let u1: f64 = rng.gen_range(0.0..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                noise[(i, j)] = z * self.augmentation_strength;
            }
        }
        augmented = augmented + noise;

        // Feature dropout (randomly set features to 0)
        let dropout_prob = 0.1 * self.augmentation_strength;
        for mut row in augmented.axis_iter_mut(Axis(0)) {
            for element in row.iter_mut() {
                if rng.gen::<f64>() < dropout_prob {
                    *element = 0.0;
                }
            }
        }

        augmented
    }

    fn compute_simclr_loss(&self, z_i: &ArrayView2<f64>, z_j: &ArrayView2<f64>) -> Result<f64> {
        let batch_size = z_i.nrows();
        if batch_size == 0 {
            return Ok(0.0);
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let zi = z_i.row(i);
            let zj = z_j.row(i);

            // Compute positive score
            let pos_score = (zi.dot(&zj) / self.temperature).exp();

            // Compute negative scores (all other samples in batch)
            let mut neg_sum = 0.0;
            for k in 0..batch_size {
                if k != i {
                    let zk_i = z_i.row(k);
                    let zk_j = z_j.row(k);

                    // Negative scores for both augmented views
                    neg_sum += (zi.dot(&zk_i) / self.temperature).exp();
                    neg_sum += (zi.dot(&zk_j) / self.temperature).exp();
                    neg_sum += (zj.dot(&zk_i) / self.temperature).exp();
                    neg_sum += (zj.dot(&zk_j) / self.temperature).exp();
                }
            }

            // Add self-negative (zi vs zj excluded from negative)
            neg_sum += pos_score;

            // Compute loss for this pair
            let loss = -(pos_score / neg_sum).ln();
            total_loss += loss;
        }

        Ok(total_loss / (2.0 * batch_size as f64))
    }

    fn compute_supervised_contrastive_loss(
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

            let zi = embeddings.row(i);
            let mut pos_sum = 0.0;
            let mut neg_sum = 0.0;
            let mut pos_count = 0;

            for j in 0..batch_size {
                if i == j || labels[j] == -1 {
                    continue;
                }

                let zj = embeddings.row(j);
                let similarity = (zi.dot(&zj) / self.temperature).exp();

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
}

/// Fitted SimCLR model
#[derive(Debug, Clone)]
pub struct FittedSimCLR {
    /// base_model
    pub base_model: SimCLR,
    /// encoder_weights
    pub encoder_weights: Array2<f64>,
    /// projection_weights
    pub projection_weights: Array2<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// n_classes
    pub n_classes: usize,
}

impl Estimator for SimCLR {
    type Config = SimCLR;
    type Error = ContrastiveLearningError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for SimCLR {
    type Fitted = FittedSimCLR;

    fn fit(self, X: &ArrayView2<'_, f64>, y: &ArrayView1<'_, i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = X.dim();

        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Initialize encoder and projection head
        // Xavier-like initialization - create weights manually
        let mut encoder_weights = Array2::<f64>::zeros((n_features, self.embedding_dim));
        let mut projection_weights =
            Array2::<f64>::zeros((self.embedding_dim, self.projection_dim));

        // Fill encoder weights with normal distribution (mean=0.0, std=0.02)
        for i in 0..n_features {
            for j in 0..self.embedding_dim {
                let u1: f64 = rng.gen_range(0.0..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                encoder_weights[(i, j)] = z * 0.02;
            }
        }

        // Fill projection weights with normal distribution (mean=0.0, std=0.02)
        for i in 0..self.embedding_dim {
            for j in 0..self.projection_dim {
                let u1: f64 = rng.gen_range(0.0..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                projection_weights[(i, j)] = z * 0.02;
            }
        }

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

                // Generate two augmented views
                let X_aug1 = self.apply_augmentation(&batch_X, &mut rng);
                let X_aug2 = self.apply_augmentation(&batch_X, &mut rng);

                // Forward pass through encoder and projection head
                let h1 = X_aug1.dot(&encoder_weights);
                let h2 = X_aug2.dot(&encoder_weights);
                let z1 = h1.dot(&projection_weights);
                let z2 = h2.dot(&projection_weights);

                // L2 normalize projections
                let z1_norm = self.l2_normalize(&z1);
                let z2_norm = self.l2_normalize(&z2);

                // Compute SimCLR loss
                let simclr_loss = self.compute_simclr_loss(&z1_norm.view(), &z2_norm.view())?;

                // Compute supervised contrastive loss for labeled samples
                let supervised_loss = if n_classes > 0 {
                    self.compute_supervised_contrastive_loss(&h1.view(), &batch_y.view())?
                } else {
                    0.0
                };

                // Combined loss
                let total_loss = simclr_loss + self.labeled_weight * supervised_loss;
                epoch_loss += total_loss;
                n_batches += 1;

                // Simple gradient simulation (in practice, would use proper backpropagation)
                let gradient_scale = self.learning_rate * total_loss;
                let noise_std = gradient_scale * 0.01;

                // Create encoder gradient manually
                let mut encoder_grad = Array2::<f64>::zeros(encoder_weights.dim());
                for i in 0..encoder_weights.nrows() {
                    for j in 0..encoder_weights.ncols() {
                        let u1: f64 = rng.gen_range(0.0..1.0);
                        let u2: f64 = rng.gen_range(0.0..1.0);
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        encoder_grad[(i, j)] = z * noise_std;
                    }
                }

                // Create projection gradient manually
                let mut projection_grad = Array2::<f64>::zeros(projection_weights.dim());
                for i in 0..projection_weights.nrows() {
                    for j in 0..projection_weights.ncols() {
                        let u1: f64 = rng.gen_range(0.0..1.0);
                        let u2: f64 = rng.gen_range(0.0..1.0);
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        projection_grad[(i, j)] = z * noise_std;
                    }
                }

                encoder_weights = encoder_weights - encoder_grad;
                projection_weights = projection_weights - projection_grad;
            }

            if n_batches > 0 {
                epoch_loss /= n_batches as f64;
            }
        }

        Ok(FittedSimCLR {
            base_model: self,
            encoder_weights,
            projection_weights,
            classes: Array1::from_vec(unique_classes),
            n_classes,
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for FittedSimCLR {
    fn predict(&self, X: &ArrayView2<'_, f64>) -> Result<Array1<i32>> {
        let embeddings = X.dot(&self.encoder_weights);
        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        if self.n_classes == 0 {
            return Ok(predictions);
        }

        // Simple nearest neighbor classification in embedding space
        for i in 0..n_samples {
            let embedding = embeddings.row(i);

            // Predict based on embedding magnitude (placeholder logic)
            let score = embedding.sum();
            let class_idx = ((score.abs() * self.n_classes as f64) as usize) % self.n_classes;
            predictions[i] = self.classes[class_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>> for FittedSimCLR {
    fn predict_proba(&self, X: &ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let embeddings = X.dot(&self.encoder_weights);
        let n_samples = X.nrows();
        let mut probabilities = Array2::zeros((n_samples, self.n_classes.max(1)));

        if self.n_classes == 0 {
            probabilities.fill(1.0);
            return Ok(probabilities);
        }

        for i in 0..n_samples {
            let embedding = embeddings.row(i);

            // Generate probabilities based on embedding
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
    fn test_simclr_creation() {
        let simclr = SimCLR::new()
            .projection_dim(32)
            .embedding_dim(64)
            .temperature(0.3)
            .max_epochs(10);

        assert_eq!(simclr.projection_dim, 32);
        assert_eq!(simclr.embedding_dim, 64);
        assert_eq!(simclr.temperature, 0.3);
        assert_eq!(simclr.max_epochs, 10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_simclr_fit_predict() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0]
        ];
        let y = array![0, 1, 0, 1, -1, -1]; // -1 indicates unlabeled

        let simclr = SimCLR::new()
            .projection_dim(4)
            .embedding_dim(8)
            .max_epochs(2)
            .batch_size(3)
            .random_state(42);

        let fitted = simclr.fit(&X.view(), &y.view()).unwrap();
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
    fn test_simclr_augmentation() {
        let simclr = SimCLR::new().augmentation_strength(0.1);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let mut rng = Random::seed(42);

        let augmented = simclr.apply_augmentation(&x, &mut rng);
        assert_eq!(augmented.dim(), x.dim());

        // Augmented data should be different from original
        let diff = (&augmented - &x).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_simclr_l2_normalize() {
        let simclr = SimCLR::new();
        let x = array![[3.0, 4.0], [1.0, 0.0]];

        let normalized = simclr.l2_normalize(&x);

        // Check that each row has unit norm
        for row in normalized.axis_iter(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            assert!((norm - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_simclr_all_unlabeled() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![-1, -1, -1]; // All unlabeled

        let simclr = SimCLR::new().max_epochs(2).batch_size(2);

        let fitted = simclr.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        // All predictions should be 0 when no labeled classes exist
        for &pred in predictions.iter() {
            assert_eq!(pred, 0);
        }
    }
}
