//! Supervised Contrastive Learning implementation for semi-supervised scenarios

use super::{ContrastiveLearningError, *};
use scirs2_core::random::Rng;

/// Supervised Contrastive Learning for semi-supervised scenarios
///
/// This method extends contrastive learning to utilize both labeled and unlabeled data
/// by pulling together samples from the same class while pushing apart samples from different classes.
#[derive(Debug, Clone)]
pub struct SupervisedContrastiveLearning {
    /// embedding_dim
    pub embedding_dim: usize,
    /// temperature
    pub temperature: f64,
    /// learning_rate
    pub learning_rate: f64,
    /// batch_size
    pub batch_size: usize,
    /// max_epochs
    pub max_epochs: usize,
    /// augmentation_strength
    pub augmentation_strength: f64,
    /// labeled_weight
    pub labeled_weight: f64,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for SupervisedContrastiveLearning {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            temperature: 0.07,
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            augmentation_strength: 0.5,
            labeled_weight: 2.0,
            random_state: None,
        }
    }
}

impl SupervisedContrastiveLearning {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
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

    pub fn augmentation_strength(mut self, augmentation_strength: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&augmentation_strength) {
            return Err(ContrastiveLearningError::InvalidAugmentationStrength(
                augmentation_strength,
            )
            .into());
        }
        self.augmentation_strength = augmentation_strength;
        Ok(self)
    }

    pub fn labeled_weight(mut self, labeled_weight: f64) -> Self {
        self.labeled_weight = labeled_weight;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    fn augment_data<R>(&self, X: &ArrayView2<f64>, rng: &mut Random<R>) -> Result<Array2<f64>>
    where
        R: Rng,
    {
        let (n_samples, n_features) = X.dim();
        let mut augmented = X.to_owned();

        // Gaussian noise augmentation - create noise manually
        let noise_std = self.augmentation_strength * 0.1;
        let mut noise = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                // Generate normal distributed random number
                let u1: f64 = rng.gen_range(0.0..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                noise[(i, j)] = z * noise_std;
            }
        }
        augmented = augmented + noise;

        Ok(augmented)
    }

    fn compute_supervised_contrastive_loss(
        &self,
        embeddings: &ArrayView2<f64>,
        labels: &ArrayView1<i32>,
    ) -> Result<f64> {
        let n_samples = embeddings.dim().0;
        let mut total_loss = 0.0;
        let mut n_labeled = 0;

        for i in 0..n_samples {
            if labels[i] == -1 {
                continue; // Skip unlabeled samples
            }

            let anchor = embeddings.row(i);
            let anchor_label = labels[i];

            let mut positive_scores = Vec::new();
            let mut negative_scores = Vec::new();

            for j in 0..n_samples {
                if i == j {
                    continue;
                }

                let sample = embeddings.row(j);
                let score = anchor.dot(&sample) / self.temperature;

                if labels[j] == anchor_label && labels[j] != -1 {
                    positive_scores.push(score);
                } else if labels[j] != -1 {
                    negative_scores.push(score);
                }
            }

            if positive_scores.is_empty() {
                continue;
            }

            // Compute supervised contrastive loss
            let max_score = positive_scores
                .iter()
                .chain(negative_scores.iter())
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            let mut pos_exp_sum = 0.0;
            for &score in positive_scores.iter() {
                pos_exp_sum += (score - max_score).exp();
            }

            let mut all_exp_sum = pos_exp_sum;
            for &score in negative_scores.iter() {
                all_exp_sum += (score - max_score).exp();
            }

            if all_exp_sum > 0.0 {
                let loss = -(pos_exp_sum / all_exp_sum).ln();
                total_loss += loss;
                n_labeled += 1;
            }
        }

        if n_labeled > 0 {
            Ok(total_loss / n_labeled as f64)
        } else {
            Ok(0.0)
        }
    }
}

/// Fitted Supervised Contrastive Learning model
#[derive(Debug, Clone)]
pub struct FittedSupervisedContrastiveLearning {
    /// base_model
    pub base_model: SupervisedContrastiveLearning,
    /// encoder_weights
    pub encoder_weights: Array2<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// n_classes
    pub n_classes: usize,
    /// class_centroids
    pub class_centroids: Array2<f64>,
}

impl Estimator for SupervisedContrastiveLearning {
    type Config = SupervisedContrastiveLearning;
    type Error = ContrastiveLearningError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for SupervisedContrastiveLearning {
    type Fitted = FittedSupervisedContrastiveLearning;

    #[allow(non_snake_case)]
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

        // Initialize encoder
        // Initialize encoder weights manually
        let mut encoder_weights = Array2::<f64>::zeros((n_features, self.embedding_dim));
        for i in 0..n_features {
            for j in 0..self.embedding_dim {
                // Generate normal distributed random number (mean=0.0, std=0.1)
                let u1: f64 = rng.gen_range(0.0..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                encoder_weights[(i, j)] = z * 0.1;
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
            // Augment data
            let X_aug = self.augment_data(X, &mut rng)?;

            // Encode data
            let embeddings = X_aug.dot(&encoder_weights);

            // Compute supervised contrastive loss
            let loss = self.compute_supervised_contrastive_loss(&embeddings.view(), y)?;

            // Simple gradient update (placeholder)
            let gradient_scale = self.learning_rate * loss;
            // Create gradient noise manually
            let noise_std = gradient_scale * 0.1;
            let mut encoder_grad = Array2::<f64>::zeros(encoder_weights.dim());
            for i in 0..encoder_weights.nrows() {
                for j in 0..encoder_weights.ncols() {
                    let u1: f64 = rng.gen_range(0.0..1.0);
                    let u2: f64 = rng.gen_range(0.0..1.0);
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    encoder_grad[(i, j)] = z * noise_std;
                }
            }
            encoder_weights = encoder_weights - encoder_grad;

            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, loss);
            }
        }

        // Compute class centroids
        let final_embeddings = X.dot(&encoder_weights);

        let mut class_centroids = Array2::zeros((n_classes, self.embedding_dim));
        let mut class_counts = vec![0; n_classes];

        for i in 0..n_samples {
            if y[i] != -1 {
                if let Some(class_idx) = unique_classes.iter().position(|&c| c == y[i]) {
                    for j in 0..self.embedding_dim {
                        class_centroids[[class_idx, j]] += final_embeddings[[i, j]];
                    }
                    class_counts[class_idx] += 1;
                }
            }
        }

        // Normalize centroids
        for class_idx in 0..n_classes {
            if class_counts[class_idx] > 0 {
                for j in 0..self.embedding_dim {
                    class_centroids[[class_idx, j]] /= class_counts[class_idx] as f64;
                }
            }
        }

        Ok(FittedSupervisedContrastiveLearning {
            base_model: self.clone(),
            encoder_weights,
            classes: Array1::from_vec(unique_classes),
            n_classes,
            class_centroids,
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for FittedSupervisedContrastiveLearning {
    fn predict(&self, X: &ArrayView2<'_, f64>) -> Result<Array1<i32>> {
        let embeddings = X.dot(&self.encoder_weights);

        let n_samples = X.dim().0;
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let embedding = embeddings.row(i);
            let mut best_class = self.classes[0];
            let mut best_distance = f64::INFINITY;

            for (class_idx, &class) in self.classes.iter().enumerate() {
                let centroid = self.class_centroids.row(class_idx);
                let distance = embedding
                    .iter()
                    .zip(centroid.iter())
                    .map(|(e, c)| (e - c).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance < best_distance {
                    best_distance = distance;
                    best_class = class;
                }
            }

            predictions[i] = best_class;
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>> for FittedSupervisedContrastiveLearning {
    fn predict_proba(&self, X: &ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let embeddings = X.dot(&self.encoder_weights);

        let n_samples = X.dim().0;
        let mut probabilities = Array2::zeros((n_samples, self.n_classes));

        for i in 0..n_samples {
            let embedding = embeddings.row(i);
            let mut distances = Vec::new();

            for class_idx in 0..self.n_classes {
                let centroid = self.class_centroids.row(class_idx);
                let distance = embedding
                    .iter()
                    .zip(centroid.iter())
                    .map(|(e, c)| (e - c).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances.push(-distance); // Negative distance for softmax
            }

            // Softmax normalization
            let max_distance = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_distances: Vec<f64> = distances
                .iter()
                .map(|&d| (d - max_distance).exp())
                .collect();
            let sum_exp: f64 = exp_distances.iter().sum();

            for (j, &exp_dist) in exp_distances.iter().enumerate() {
                probabilities[[i, j]] = exp_dist / sum_exp;
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
    fn test_supervised_contrastive_learning_creation() {
        let scl = SupervisedContrastiveLearning::new()
            .embedding_dim(32)
            .temperature(0.05)
            .unwrap()
            .augmentation_strength(0.3)
            .unwrap()
            .labeled_weight(3.0)
            .random_state(42);

        assert_eq!(scl.embedding_dim, 32);
        assert_eq!(scl.temperature, 0.05);
        assert_eq!(scl.augmentation_strength, 0.3);
        assert_eq!(scl.labeled_weight, 3.0);
        assert_eq!(scl.random_state, Some(42));
    }

    #[test]
    fn test_supervised_contrastive_learning_invalid_augmentation() {
        let result = SupervisedContrastiveLearning::new().augmentation_strength(1.5);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_supervised_contrastive_learning_fit_predict() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0]
        ];
        let y = array![0, 1, 0, 1, -1, -1];

        let scl = SupervisedContrastiveLearning::new()
            .embedding_dim(4)
            .max_epochs(2)
            .batch_size(3)
            .unwrap()
            .random_state(42);

        let fitted = scl.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 6);
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
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_insufficient_labeled_samples() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![-1, -1]; // All unlabeled

        let scl = SupervisedContrastiveLearning::new();
        let result = scl.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }
}
