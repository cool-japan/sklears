//! Prototypical Networks implementation

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Prototypical Networks for Few-Shot Learning
///
/// Prototypical Networks learn a metric space where classification can be performed
/// by computing distances to prototype representations of each class. The prototypes
/// are the mean of the support examples for each class in an embedding space.
///
/// The method is particularly effective for few-shot learning scenarios where
/// only a few labeled examples are available per class.
///
/// # Parameters
///
/// * `embedding_dim` - Dimensionality of the embedding space
/// * `hidden_layers` - Hidden layer dimensions for the embedding network
/// * `distance_metric` - Distance metric to use ('euclidean', 'cosine', 'manhattan')
/// * `learning_rate` - Learning rate for embedding network training
/// * `n_episodes` - Number of training episodes
/// * `n_way` - Number of classes per episode
/// * `n_shot` - Number of support examples per class
/// * `n_query` - Number of query examples per class
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::PrototypicalNetworks;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![
///     [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
///     [1.1, 2.1], [2.1, 3.1], [3.1, 4.1], [4.1, 5.1]
/// ];
/// let y = array![0, 1, 0, 1, 0, 1, 0, 1];
///
/// let proto_net = PrototypicalNetworks::new()
///     .embedding_dim(32)
///     .n_way(2)
///     .n_shot(1)
///     .n_query(3);
/// let fitted = proto_net.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PrototypicalNetworks<S = Untrained> {
    state: S,
    embedding_dim: usize,
    hidden_layers: Vec<usize>,
    distance_metric: String,
    learning_rate: f64,
    n_episodes: usize,
    n_way: usize,
    n_shot: usize,
    n_query: usize,
    temperature: f64,
}

impl PrototypicalNetworks<Untrained> {
    /// Create a new PrototypicalNetworks instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            embedding_dim: 64,
            hidden_layers: vec![128, 64],
            distance_metric: "euclidean".to_string(),
            learning_rate: 0.001,
            n_episodes: 100,
            n_way: 5,
            n_shot: 1,
            n_query: 15,
            temperature: 1.0,
        }
    }

    /// Set the embedding dimensionality
    pub fn embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set the hidden layer dimensions
    pub fn hidden_layers(mut self, hidden_layers: Vec<usize>) -> Self {
        self.hidden_layers = hidden_layers;
        self
    }

    /// Set the distance metric
    pub fn distance_metric(mut self, metric: String) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of training episodes
    pub fn n_episodes(mut self, n_episodes: usize) -> Self {
        self.n_episodes = n_episodes;
        self
    }

    /// Set the number of classes per episode (N-way)
    pub fn n_way(mut self, n_way: usize) -> Self {
        self.n_way = n_way;
        self
    }

    /// Set the number of support examples per class (N-shot)
    pub fn n_shot(mut self, n_shot: usize) -> Self {
        self.n_shot = n_shot;
        self
    }

    /// Set the number of query examples per class
    pub fn n_query(mut self, n_query: usize) -> Self {
        self.n_query = n_query;
        self
    }

    /// Set the temperature parameter for softmax
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Compute embedding for input data
    fn compute_embedding(
        &self,
        X: &Array2<f64>,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> Array2<f64> {
        let mut current = X.clone();

        for (i, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
            current = current.dot(w);

            // Add bias
            for mut row in current.axis_iter_mut(Axis(0)) {
                for (j, &bias_val) in b.iter().enumerate() {
                    row[j] += bias_val;
                }
            }

            // Apply ReLU activation (except for last layer)
            if i < weights.len() - 1 {
                current.mapv_inplace(|x| x.max(0.0));
            }
        }

        current
    }

    /// Compute distance between embeddings
    fn compute_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        match self.distance_metric.as_str() {
            "euclidean" => {
                let diff = a - b;
                diff.mapv(|x| x * x).sum().sqrt()
            }
            "cosine" => {
                let dot_product = a.dot(b);
                let norm_a = a.mapv(|x| x * x).sum().sqrt();
                let norm_b = b.mapv(|x| x * x).sum().sqrt();
                1.0 - (dot_product / (norm_a * norm_b))
            }
            "manhattan" => {
                let diff = a - b;
                diff.mapv(|x| x.abs()).sum()
            }
            _ => {
                // Default to euclidean
                let diff = a - b;
                diff.mapv(|x| x * x).sum().sqrt()
            }
        }
    }

    /// Compute prototypes for each class
    fn compute_prototypes(
        &self,
        support_embeddings: &Array2<f64>,
        support_labels: &Array1<i32>,
        classes: &[i32],
    ) -> Array2<f64> {
        let n_classes = classes.len();
        let embedding_dim = support_embeddings.ncols();
        let mut prototypes = Array2::zeros((n_classes, embedding_dim));

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let mut class_embeddings = Vec::new();

            for (sample_idx, &label) in support_labels.iter().enumerate() {
                if label == class_label {
                    class_embeddings.push(support_embeddings.row(sample_idx).to_owned());
                }
            }

            if !class_embeddings.is_empty() {
                // Compute mean embedding as prototype
                for dim in 0..embedding_dim {
                    let mean_val: f64 = class_embeddings.iter().map(|emb| emb[dim]).sum::<f64>()
                        / class_embeddings.len() as f64;
                    prototypes[[class_idx, dim]] = mean_val;
                }
            }
        }

        prototypes
    }

    /// Apply softmax to distances
    fn softmax_distances(&self, distances: &Array1<f64>) -> Array1<f64> {
        let scaled_distances = distances.mapv(|d| -d / self.temperature);
        let max_dist = scaled_distances
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        let exp_distances = scaled_distances.mapv(|d| (d - max_dist).exp());
        let sum_exp = exp_distances.sum();

        exp_distances.mapv(|x| x / sum_exp)
    }

    /// Sample episode for training
    fn sample_episode(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        classes: &[i32],
    ) -> SklResult<(Array2<f64>, Array1<i32>, Array2<f64>, Array1<i32>)> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Group samples by class
        let mut class_samples: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_samples.entry(label).or_default().push(i);
        }

        // Check if we have enough samples per class
        for &class_label in classes {
            if let Some(samples) = class_samples.get(&class_label) {
                if samples.len() < self.n_shot + self.n_query {
                    return Err(SklearsError::InvalidInput(format!(
                        "Not enough samples for class {}: need {}, have {}",
                        class_label,
                        self.n_shot + self.n_query,
                        samples.len()
                    )));
                }
            } else {
                return Err(SklearsError::InvalidInput(format!(
                    "Class {} not found in data",
                    class_label
                )));
            }
        }

        // Sample support and query sets
        let total_support = self.n_way * self.n_shot;
        let total_query = self.n_way * self.n_query;

        let mut support_X = Array2::zeros((total_support, n_features));
        let mut support_y = Array1::zeros(total_support);
        let mut query_X = Array2::zeros((total_query, n_features));
        let mut query_y = Array1::zeros(total_query);

        let mut support_idx = 0;
        let mut query_idx = 0;

        for (class_idx, &class_label) in classes.iter().take(self.n_way).enumerate() {
            if let Some(samples) = class_samples.get(&class_label) {
                // Randomly sample from available samples (simplified - just take first few)
                let selected_samples: Vec<usize> = samples
                    .iter()
                    .take(self.n_shot + self.n_query)
                    .cloned()
                    .collect();

                // Support set
                for i in 0..self.n_shot {
                    let sample_idx = selected_samples[i];
                    support_X.row_mut(support_idx).assign(&X.row(sample_idx));
                    support_y[support_idx] = class_idx as i32; // Use episode-specific class indices
                    support_idx += 1;
                }

                // Query set
                for i in self.n_shot..self.n_shot + self.n_query {
                    let sample_idx = selected_samples[i];
                    query_X.row_mut(query_idx).assign(&X.row(sample_idx));
                    query_y[query_idx] = class_idx as i32; // Use episode-specific class indices
                    query_idx += 1;
                }
            }
        }

        Ok((support_X, support_y, query_X, query_y))
    }
}

impl Default for PrototypicalNetworks<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for PrototypicalNetworks<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for PrototypicalNetworks<Untrained> {
    type Fitted = PrototypicalNetworks<PrototypicalNetworksTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        let (n_samples, n_features) = X.dim();

        // Get unique classes
        let mut classes = std::collections::HashSet::new();
        for &label in y.iter() {
            if label != -1 {
                classes.insert(label);
            }
        }
        let classes: Vec<i32> = classes.into_iter().collect();

        if classes.len() < self.n_way {
            return Err(SklearsError::InvalidInput(format!(
                "Need at least {} classes for {}-way classification, found {}",
                self.n_way,
                self.n_way,
                classes.len()
            )));
        }

        // Initialize embedding network weights
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend(&self.hidden_layers);
        layer_sizes.push(self.embedding_dim);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let in_size = layer_sizes[i];
            let out_size = layer_sizes[i + 1];

            // Xavier initialization
            let scale = (2.0 / (in_size + out_size) as f64).sqrt();
            let mut w = Array2::zeros((in_size, out_size));
            let b = Array1::zeros(out_size);

            // Simple initialization
            for i in 0..in_size {
                for j in 0..out_size {
                    w[[i, j]] = scale * ((i + j) as f64 * 0.1).sin();
                }
            }

            weights.push(w);
            biases.push(b);
        }

        // Episodic training
        for episode in 0..self.n_episodes {
            // Sample an episode
            let episode_classes: Vec<i32> = classes.iter().take(self.n_way).cloned().collect();

            let (support_X, support_y, query_X, query_y) =
                self.sample_episode(&X, &y, &episode_classes)?;

            // Forward pass: compute embeddings
            let support_embeddings = self.compute_embedding(&support_X, &weights, &biases);
            let query_embeddings = self.compute_embedding(&query_X, &weights, &biases);

            // Compute prototypes
            let episode_class_indices: Vec<i32> = (0..self.n_way as i32).collect();
            let prototypes =
                self.compute_prototypes(&support_embeddings, &support_y, &episode_class_indices);

            // Compute distances and probabilities for query set
            let n_query_samples = query_embeddings.nrows();
            let mut total_loss = 0.0;

            for query_idx in 0..n_query_samples {
                let query_embedding = query_embeddings.row(query_idx);
                let true_class = query_y[query_idx] as usize;

                // Skip if true_class is out of bounds
                if true_class >= self.n_way {
                    continue;
                }

                // Compute distances to all prototypes
                let mut distances = Array1::zeros(self.n_way);
                for class_idx in 0..self.n_way {
                    let prototype = prototypes.row(class_idx);
                    distances[class_idx] =
                        self.compute_distance(&query_embedding.to_owned(), &prototype.to_owned());
                }

                // Convert to probabilities
                let probabilities = self.softmax_distances(&distances);

                // Cross-entropy loss
                let prob = probabilities[true_class].max(1e-10);
                total_loss -= prob.ln();

                // Simple gradient update (simplified for demonstration)
                let lr = self.learning_rate / (episode + 1) as f64;

                // Update last layer weights (simplified gradient)
                if let (Some(last_w), Some(last_b)) = (weights.last_mut(), biases.last_mut()) {
                    let max_features = query_X.ncols().min(last_w.nrows());
                    for i in 0..max_features {
                        for j in 0..last_w.ncols() {
                            let grad_w =
                                (probabilities[true_class] - 1.0) * query_X[[query_idx, i]];
                            last_w[[i, j]] -= lr * grad_w;
                        }
                    }

                    for j in 0..last_b.len() {
                        let grad_b = probabilities[true_class] - 1.0;
                        last_b[j] -= lr * grad_b;
                    }
                }
            }

            // Print training progress occasionally
            if episode % 20 == 0 {
                let avg_loss = total_loss / n_query_samples as f64;
                // Could log progress here if needed
                let _ = avg_loss; // Suppress unused variable warning
            }
        }

        Ok(PrototypicalNetworks {
            state: PrototypicalNetworksTrained {
                weights,
                biases,
                classes: Array1::from(classes),
                prototypes: Array2::zeros((1, 1)), // Will be computed during prediction
            },
            embedding_dim: self.embedding_dim,
            hidden_layers: self.hidden_layers,
            distance_metric: self.distance_metric,
            learning_rate: self.learning_rate,
            n_episodes: self.n_episodes,
            n_way: self.n_way,
            n_shot: self.n_shot,
            n_query: self.n_query,
            temperature: self.temperature,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for PrototypicalNetworks<PrototypicalNetworksTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            let max_idx = probabilities
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            predictions[i] = self.state.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for PrototypicalNetworks<PrototypicalNetworksTrained>
{
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();

        // For prediction, we need support examples to compute prototypes
        // This is a limitation of the current implementation - in practice,
        // we would store representative prototypes from training

        // For now, return uniform probabilities as a placeholder
        let mut probabilities = Array2::zeros((n_test, n_classes));
        for i in 0..n_test {
            for j in 0..n_classes {
                probabilities[[i, j]] = 1.0 / n_classes as f64;
            }
        }

        Ok(probabilities)
    }
}

/// Trained state for PrototypicalNetworks
#[derive(Debug, Clone)]
pub struct PrototypicalNetworksTrained {
    /// weights
    pub weights: Vec<Array2<f64>>,
    /// biases
    pub biases: Vec<Array1<f64>>,
    /// classes
    pub classes: Array1<i32>,
    /// prototypes
    pub prototypes: Array2<f64>,
}
