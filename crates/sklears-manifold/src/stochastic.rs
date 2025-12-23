//! Stochastic Manifold Learning Algorithms
//! This module provides stochastic manifold learning algorithms designed for
//! large datasets that cannot fit in memory or require online/streaming processing.
//! These algorithms use random sampling, mini-batch processing, and incremental
//! updates to handle massive datasets efficiently.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::{seq::SliceRandom, Rng, SeedableRng};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Fit, Transform},
    types::Float,
};
use std::collections::VecDeque;

/// Stochastic Gradient Descent Manifold Learning
///
/// This algorithm uses stochastic gradient descent to learn manifold embeddings
/// incrementally, making it suitable for very large datasets that don't fit in memory.
/// It supports various manifold learning objectives including stress minimization
/// and neighborhood preservation.
///
/// # Parameters
///
/// * `n_components` - Number of dimensions in the embedded space
/// * `learning_rate` - Learning rate for gradient descent
/// * `batch_size` - Size of mini-batches for processing
/// * `n_epochs` - Number of training epochs
/// * `momentum` - Momentum parameter for gradient updates
/// * `objective` - Objective function ('stress', 'neighbor_preservation', 'tsne')
/// * `n_neighbors` - Number of neighbors to consider for local objectives
/// * `perplexity` - Perplexity parameter for t-SNE objective
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_manifold::stochastic::StochasticManifoldLearning;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::{ Array1, ArrayView1};
///
/// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let sml = StochasticManifoldLearning::new(2)
///     .batch_size(2);
/// let fitted = sml.fit(&data, &()).unwrap();
/// let embedding = fitted.transform(&data).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StochasticManifoldLearning {
    n_components: usize,
    learning_rate: Float,
    batch_size: usize,
    n_epochs: usize,
    momentum: Float,
    objective: String,
    n_neighbors: usize,
    perplexity: Float,
    random_state: Option<u64>,
}

impl StochasticManifoldLearning {
    /// Create a new Stochastic Manifold Learning instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            learning_rate: 0.01,
            batch_size: 100,
            n_epochs: 50,
            momentum: 0.9,
            objective: "stress".to_string(),
            n_neighbors: 10,
            perplexity: 30.0,
            random_state: None,
        }
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the number of epochs
    pub fn n_epochs(mut self, n_epochs: usize) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Set the momentum parameter
    pub fn momentum(mut self, momentum: Float) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the objective function
    pub fn objective(mut self, objective: &str) -> Self {
        self.objective = objective.to_string();
        self
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the perplexity for t-SNE objective
    pub fn perplexity(mut self, perplexity: Float) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

/// Fitted Stochastic Manifold Learning model
#[derive(Debug, Clone)]
pub struct FittedStochasticManifoldLearning {
    n_components: usize,
    embedding: Array2<Float>,
    learning_rate: Float,
    momentum: Float,
    objective: String,
    n_neighbors: usize,
    perplexity: Float,
    training_data: Array2<Float>,
    loss_history: Vec<Float>,
}

impl Fit<Array2<Float>, ()> for StochasticManifoldLearning {
    type Fitted = FittedStochasticManifoldLearning;

    fn fit(self, data: &Array2<Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = data;
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.n_components >= n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) must be less than n_samples ({})",
                self.n_components, n_samples
            )));
        }

        if self.batch_size > n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "batch_size ({}) cannot be greater than n_samples ({})",
                self.batch_size, n_samples
            )));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen())
        };

        // Initialize embedding randomly
        let mut embedding = Array2::<Float>::zeros((n_samples, self.n_components));
        for elem in embedding.iter_mut() {
            *elem = rng.sample::<Float, _>(scirs2_core::StandardNormal) * 0.01;
        }

        // Initialize momentum variables
        let mut velocity = Array2::zeros((n_samples, self.n_components));
        let mut loss_history = Vec::new();

        // Precompute distances and neighbors if needed
        let (distances, neighbor_indices) = match self.objective.as_str() {
            "neighbor_preservation" | "tsne" => {
                let distances = self.compute_pairwise_distances(x)?;
                let neighbors = self.find_k_neighbors(&distances)?;
                (Some(distances), Some(neighbors))
            }
            _ => (None, None),
        };

        // Training loop
        for epoch in 0..self.n_epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            // Create batches
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            for batch_indices in indices.chunks(self.batch_size) {
                let batch_loss = self.process_batch(
                    x,
                    &mut embedding,
                    &mut velocity,
                    batch_indices,
                    &distances,
                    &neighbor_indices,
                    epoch,
                )?;

                epoch_loss += batch_loss;
                n_batches += 1;
            }

            let avg_loss = epoch_loss / n_batches as Float;
            loss_history.push(avg_loss);
        }

        Ok(FittedStochasticManifoldLearning {
            n_components: self.n_components,
            embedding,
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            objective: self.objective.clone(),
            n_neighbors: self.n_neighbors,
            perplexity: self.perplexity,
            training_data: x.clone(),
            loss_history,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for FittedStochasticManifoldLearning {
    fn transform(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        // For new data, we need to find embedding using the learned manifold
        // This is a simplified implementation - in practice, we'd use out-of-sample methods
        if data.nrows() != self.training_data.nrows() || data.ncols() != self.training_data.ncols()
        {
            return Err(SklearsError::InvalidInput(
                "Transform only supports the same data used for training in this implementation"
                    .to_string(),
            ));
        }

        Ok(self.embedding.clone())
    }
}

impl StochasticManifoldLearning {
    /// Process a single batch of data
    fn process_batch(
        &self,
        data: &Array2<Float>,
        embedding: &mut Array2<Float>,
        velocity: &mut Array2<Float>,
        batch_indices: &[usize],
        distances: &Option<Array2<Float>>,
        neighbor_indices: &Option<Vec<Vec<usize>>>,
        epoch: usize,
    ) -> SklResult<Float> {
        let mut batch_loss = 0.0;
        let current_lr = self.learning_rate / (1.0 + epoch as Float * 0.1);

        // Compute gradients for the batch
        let gradients =
            self.compute_gradients(data, embedding, batch_indices, distances, neighbor_indices)?;

        // Update embeddings with momentum
        for &idx in batch_indices {
            for j in 0..self.n_components {
                // Momentum update
                velocity[[idx, j]] =
                    self.momentum * velocity[[idx, j]] - current_lr * gradients[[idx, j]];
                embedding[[idx, j]] += velocity[[idx, j]];
            }
        }

        // Compute batch loss
        batch_loss =
            self.compute_batch_loss(data, embedding, batch_indices, distances, neighbor_indices)?;

        Ok(batch_loss)
    }

    /// Compute gradients for the current batch
    fn compute_gradients(
        &self,
        data: &Array2<Float>,
        embedding: &Array2<Float>,
        batch_indices: &[usize],
        distances: &Option<Array2<Float>>,
        neighbor_indices: &Option<Vec<Vec<usize>>>,
    ) -> SklResult<Array2<Float>> {
        let n_samples = data.nrows();
        let mut gradients = Array2::zeros((n_samples, self.n_components));

        match self.objective.as_str() {
            "stress" => {
                self.compute_stress_gradients(data, embedding, batch_indices, &mut gradients)?;
            }
            "neighbor_preservation" => {
                if let (Some(dists), Some(neighbors)) = (distances, neighbor_indices) {
                    self.compute_neighbor_preservation_gradients(
                        embedding,
                        batch_indices,
                        dists,
                        neighbors,
                        &mut gradients,
                    )?;
                }
            }
            "tsne" => {
                if let (Some(dists), Some(neighbors)) = (distances, neighbor_indices) {
                    self.compute_tsne_gradients(
                        embedding,
                        batch_indices,
                        dists,
                        neighbors,
                        &mut gradients,
                    )?;
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown objective: {}",
                    self.objective
                )));
            }
        }

        Ok(gradients)
    }

    /// Compute stress-based gradients
    fn compute_stress_gradients(
        &self,
        data: &Array2<Float>,
        embedding: &Array2<Float>,
        batch_indices: &[usize],
        gradients: &mut Array2<Float>,
    ) -> SklResult<()> {
        let n_samples = data.nrows();

        for &i in batch_indices {
            for j in 0..n_samples {
                if i != j {
                    // Compute high-dimensional distance
                    let high_dist = euclidean_distance(&data.row(i), &data.row(j));

                    // Compute low-dimensional distance
                    let low_dist = euclidean_distance(&embedding.row(i), &embedding.row(j));

                    if low_dist > 1e-10 {
                        let factor = 2.0 * (low_dist - high_dist) / low_dist;

                        for k in 0..self.n_components {
                            let diff = embedding[[i, k]] - embedding[[j, k]];
                            gradients[[i, k]] += factor * diff;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute neighbor preservation gradients
    fn compute_neighbor_preservation_gradients(
        &self,
        embedding: &Array2<Float>,
        batch_indices: &[usize],
        distances: &Array2<Float>,
        neighbor_indices: &Vec<Vec<usize>>,
        gradients: &mut Array2<Float>,
    ) -> SklResult<()> {
        for &i in batch_indices {
            for &j in &neighbor_indices[i] {
                if i != j {
                    let high_dist = distances[[i, j]];
                    let low_dist = euclidean_distance(&embedding.row(i), &embedding.row(j));

                    if low_dist > 1e-10 {
                        let factor = 2.0 * (low_dist - high_dist) / low_dist;

                        for k in 0..self.n_components {
                            let diff = embedding[[i, k]] - embedding[[j, k]];
                            gradients[[i, k]] += factor * diff;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute t-SNE-style gradients
    fn compute_tsne_gradients(
        &self,
        embedding: &Array2<Float>,
        batch_indices: &[usize],
        distances: &Array2<Float>,
        neighbor_indices: &Vec<Vec<usize>>,
        gradients: &mut Array2<Float>,
    ) -> SklResult<()> {
        let n_samples = embedding.nrows();

        // Compute high-dimensional probabilities (simplified)
        for &i in batch_indices {
            let mut total_prob = 0.0;
            let mut high_probs = Vec::new();

            for &j in &neighbor_indices[i] {
                if i != j {
                    let dist_sq = distances[[i, j]] * distances[[i, j]];
                    let prob = (-dist_sq / (2.0 * self.perplexity)).exp();
                    high_probs.push((j, prob));
                    total_prob += prob;
                }
            }

            // Normalize probabilities
            for (j, prob) in &mut high_probs {
                *prob /= total_prob;
            }

            // Compute low-dimensional probabilities and gradients
            for (j, high_prob) in high_probs {
                let low_dist_sq = euclidean_distance_squared(&embedding.row(i), &embedding.row(j));
                let low_prob = 1.0 / (1.0 + low_dist_sq);

                let factor = 4.0 * (high_prob - low_prob) * low_prob;

                for k in 0..self.n_components {
                    let diff = embedding[[i, k]] - embedding[[j, k]];
                    gradients[[i, k]] += factor * diff;
                }
            }
        }

        Ok(())
    }

    /// Compute batch loss
    fn compute_batch_loss(
        &self,
        data: &Array2<Float>,
        embedding: &Array2<Float>,
        batch_indices: &[usize],
        distances: &Option<Array2<Float>>,
        neighbor_indices: &Option<Vec<Vec<usize>>>,
    ) -> SklResult<Float> {
        let mut loss = 0.0;
        let batch_size = batch_indices.len();

        match self.objective.as_str() {
            "stress" => {
                for &i in batch_indices {
                    for j in 0..data.nrows() {
                        if i != j {
                            let high_dist = euclidean_distance(&data.row(i), &data.row(j));
                            let low_dist = euclidean_distance(&embedding.row(i), &embedding.row(j));
                            let diff = high_dist - low_dist;
                            loss += diff * diff;
                        }
                    }
                }
                loss /= batch_size as Float;
            }
            "neighbor_preservation" => {
                if let (Some(dists), Some(neighbors)) = (distances, neighbor_indices) {
                    for &i in batch_indices {
                        for &j in &neighbors[i] {
                            if i != j {
                                let high_dist = dists[[i, j]];
                                let low_dist =
                                    euclidean_distance(&embedding.row(i), &embedding.row(j));
                                let diff = high_dist - low_dist;
                                loss += diff * diff;
                            }
                        }
                    }
                    loss /= batch_size as Float;
                }
            }
            "tsne" => {
                // Simplified t-SNE loss computation
                loss =
                    self.compute_tsne_loss(embedding, batch_indices, distances, neighbor_indices)?;
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown objective: {}",
                    self.objective
                )));
            }
        }

        Ok(loss)
    }

    /// Compute t-SNE loss
    fn compute_tsne_loss(
        &self,
        embedding: &Array2<Float>,
        batch_indices: &[usize],
        distances: &Option<Array2<Float>>,
        neighbor_indices: &Option<Vec<Vec<usize>>>,
    ) -> SklResult<Float> {
        let mut loss = 0.0;

        if let (Some(dists), Some(neighbors)) = (distances, neighbor_indices) {
            for &i in batch_indices {
                for &j in &neighbors[i] {
                    if i != j {
                        let high_dist_sq = dists[[i, j]] * dists[[i, j]];
                        let high_prob = (-high_dist_sq / (2.0 * self.perplexity)).exp();

                        let low_dist_sq =
                            euclidean_distance_squared(&embedding.row(i), &embedding.row(j));
                        let low_prob = 1.0 / (1.0 + low_dist_sq);

                        if high_prob > 1e-10 && low_prob > 1e-10 {
                            loss += high_prob * (high_prob / low_prob).ln();
                        }
                    }
                }
            }
        }

        Ok(loss)
    }

    /// Compute pairwise distances
    fn compute_pairwise_distances(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = euclidean_distance(&data.row(i), &data.row(j));
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        Ok(distances)
    }

    /// Find k-nearest neighbors
    fn find_k_neighbors(&self, distances: &Array2<Float>) -> SklResult<Vec<Vec<usize>>> {
        let n_samples = distances.nrows();
        let mut neighbors = Vec::new();

        for i in 0..n_samples {
            let mut dist_indices: Vec<(Float, usize)> = (0..n_samples)
                .filter(|&j| i != j)
                .map(|j| (distances[[i, j]], j))
                .collect();

            dist_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let k_neighbors: Vec<usize> = dist_indices
                .into_iter()
                .take(self.n_neighbors)
                .map(|(_, idx)| idx)
                .collect();

            neighbors.push(k_neighbors);
        }

        Ok(neighbors)
    }
}

impl FittedStochasticManifoldLearning {
    /// Get the learned embedding
    pub fn embedding(&self) -> &Array2<Float> {
        &self.embedding
    }

    /// Get the loss history during training
    pub fn loss_history(&self) -> &Vec<Float> {
        &self.loss_history
    }

    /// Perform partial fit on new data (online learning)
    pub fn partial_fit(&mut self, new_data: &Array2<Float>) -> SklResult<()> {
        // This is a simplified implementation of online learning
        // In practice, this would be more sophisticated

        let n_new_samples = new_data.nrows();
        let n_features = new_data.ncols();

        if n_features != self.training_data.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "New data has {} features, expected {}",
                n_features,
                self.training_data.ncols()
            )));
        }

        // Extend the embedding with new points
        let mut extended_embedding =
            Array2::zeros((self.embedding.nrows() + n_new_samples, self.n_components));

        // Copy existing embeddings
        extended_embedding
            .slice_mut(s![..self.embedding.nrows(), ..])
            .assign(&self.embedding);

        // Initialize new embeddings randomly
        let mut rng = StdRng::seed_from_u64(thread_rng().gen());
        for i in 0..n_new_samples {
            for j in 0..self.n_components {
                extended_embedding[[self.embedding.nrows() + i, j]] =
                    rng.sample::<Float, _>(scirs2_core::StandardNormal) * 0.01;
            }
        }

        // Update training data
        let mut extended_data =
            Array2::zeros((self.training_data.nrows() + n_new_samples, n_features));

        extended_data
            .slice_mut(s![..self.training_data.nrows(), ..])
            .assign(&self.training_data);
        extended_data
            .slice_mut(s![self.training_data.nrows().., ..])
            .assign(new_data);

        // Perform a few iterations of optimization on the new points
        let new_indices: Vec<usize> =
            (self.embedding.nrows()..extended_embedding.nrows()).collect();

        for _ in 0..10 {
            // Simplified gradient descent for new points
            match self.objective.as_str() {
                "stress" => {
                    for &i in &new_indices {
                        for j in 0..extended_data.nrows() {
                            if i != j {
                                let high_dist = euclidean_distance(
                                    &extended_data.row(i),
                                    &extended_data.row(j),
                                );
                                let low_dist = euclidean_distance(
                                    &extended_embedding.row(i),
                                    &extended_embedding.row(j),
                                );

                                if low_dist > 1e-10 {
                                    let factor = 2.0 * (low_dist - high_dist) / low_dist
                                        * self.learning_rate;

                                    for k in 0..self.n_components {
                                        let diff =
                                            extended_embedding[[i, k]] - extended_embedding[[j, k]];
                                        extended_embedding[[i, k]] -= factor * diff;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {
                    // Other objectives would be implemented similarly
                }
            }
        }

        self.embedding = extended_embedding;
        self.training_data = extended_data;

        Ok(())
    }
}

/// Streaming Manifold Learning
///
/// This algorithm processes data in a streaming fashion, maintaining a fixed-size
/// buffer and updating the manifold embedding incrementally as new data arrives.
/// It's designed for applications where data arrives continuously and memory is limited.
///
/// # Parameters
///
/// * `n_components` - Number of dimensions in the embedded space
/// * `buffer_size` - Maximum number of points to keep in memory
/// * `learning_rate` - Learning rate for incremental updates
/// * `decay_factor` - Factor for decaying old information
/// * `update_frequency` - How often to update the embedding
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::stochastic::StreamingManifoldLearning;
/// use scirs2_core::ndarray::array;
///
/// let mut streaming = StreamingManifoldLearning::new(2, 100);
///
/// let batch1 = array![[1.0, 2.0], [3.0, 4.0]];
/// streaming.partial_fit(&batch1).unwrap();
///
/// let batch2 = array![[5.0, 6.0], [7.0, 8.0]];
/// streaming.partial_fit(&batch2).unwrap();
///
/// let embedding = streaming.get_embedding();
/// ```
#[derive(Debug, Clone)]
pub struct StreamingManifoldLearning {
    n_components: usize,
    buffer_size: usize,
    learning_rate: Float,
    decay_factor: Float,
    update_frequency: usize,
    random_state: Option<u64>,

    // Internal state
    data_buffer: VecDeque<Array1<Float>>,
    embedding_buffer: VecDeque<Array1<Float>>,
    update_count: usize,
    rng: StdRng,
}

impl StreamingManifoldLearning {
    /// Create a new Streaming Manifold Learning instance
    pub fn new(n_components: usize, buffer_size: usize) -> Self {
        Self {
            n_components,
            buffer_size,
            learning_rate: 0.01,
            decay_factor: 0.95,
            update_frequency: 10,
            random_state: None,
            data_buffer: VecDeque::new(),
            embedding_buffer: VecDeque::new(),
            update_count: 0,
            rng: StdRng::seed_from_u64(thread_rng().gen()),
        }
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the decay factor
    pub fn decay_factor(mut self, decay_factor: Float) -> Self {
        self.decay_factor = decay_factor;
        self
    }

    /// Set the update frequency
    pub fn update_frequency(mut self, update_frequency: usize) -> Self {
        self.update_frequency = update_frequency;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self.rng = StdRng::seed_from_u64(random_state);
        self
    }

    /// Process a new batch of data
    pub fn partial_fit(&mut self, data: &Array2<Float>) -> SklResult<()> {
        for row in data.axis_iter(Axis(0)) {
            self.add_point(row.to_owned())?;
        }

        self.update_count += data.nrows();

        // Update embedding if necessary
        if self.update_count % self.update_frequency == 0 {
            self.update_embedding()?;
        }

        Ok(())
    }

    /// Add a single point to the buffer
    fn add_point(&mut self, point: Array1<Float>) -> SklResult<()> {
        // If buffer is full, remove oldest point
        if self.data_buffer.len() >= self.buffer_size {
            self.data_buffer.pop_front();
            self.embedding_buffer.pop_front();
        }

        // Add new point to data buffer
        self.data_buffer.push_back(point);

        // Initialize embedding for new point
        let mut embedding = Array1::<Float>::zeros(self.n_components);
        for elem in embedding.iter_mut() {
            *elem = self.rng.sample::<Float, _>(scirs2_core::StandardNormal) * 0.01;
        }
        self.embedding_buffer.push_back(embedding);

        Ok(())
    }

    /// Update the manifold embedding
    fn update_embedding(&mut self) -> SklResult<()> {
        if self.data_buffer.len() < 2 {
            return Ok(());
        }

        let n_points = self.data_buffer.len();

        // Convert buffers to arrays for easier processing
        let data_array = Array2::from_shape_fn((n_points, self.data_buffer[0].len()), |(i, j)| {
            self.data_buffer[i][j]
        });

        let mut embedding_array = Array2::from_shape_fn((n_points, self.n_components), |(i, j)| {
            self.embedding_buffer[i][j]
        });

        // Perform a few steps of gradient descent
        for _ in 0..5 {
            // Compute gradients using stress minimization
            let mut gradients: Array2<Float> = Array2::zeros((n_points, self.n_components));

            for i in 0..n_points {
                for j in 0..n_points {
                    if i != j {
                        let high_dist = euclidean_distance(&data_array.row(i), &data_array.row(j));
                        let low_dist =
                            euclidean_distance(&embedding_array.row(i), &embedding_array.row(j));

                        if low_dist > 1e-10 {
                            let factor = 2.0 * (low_dist - high_dist) / low_dist;

                            for k in 0..self.n_components {
                                let diff = embedding_array[[i, k]] - embedding_array[[j, k]];
                                gradients[[i, k]] += factor * diff;
                            }
                        }
                    }
                }
            }

            // Apply gradients with decay
            for i in 0..n_points {
                for j in 0..self.n_components {
                    let age_weight = self.decay_factor.powf((n_points - i - 1) as Float);
                    embedding_array[[i, j]] -= self.learning_rate * age_weight * gradients[[i, j]];
                }
            }
        }

        // Update embedding buffer
        for (i, embedding) in self.embedding_buffer.iter_mut().enumerate() {
            for j in 0..self.n_components {
                embedding[j] = embedding_array[[i, j]];
            }
        }

        Ok(())
    }

    /// Get the current embedding as an array
    pub fn get_embedding(&self) -> Array2<Float> {
        let n_points = self.embedding_buffer.len();
        if n_points == 0 {
            return Array2::zeros((0, self.n_components));
        }

        Array2::from_shape_fn((n_points, self.n_components), |(i, j)| {
            self.embedding_buffer[i][j]
        })
    }

    /// Get the current data buffer as an array
    pub fn get_data(&self) -> Array2<Float> {
        let n_points = self.data_buffer.len();
        if n_points == 0 {
            return Array2::zeros((0, 0));
        }

        let n_features = self.data_buffer[0].len();
        Array2::from_shape_fn((n_points, n_features), |(i, j)| self.data_buffer[i][j])
    }

    /// Get the number of points currently in the buffer
    pub fn buffer_size_current(&self) -> usize {
        self.data_buffer.len()
    }
}

// Helper functions

fn euclidean_distance(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    (a - b).mapv(|x| x * x).sum().sqrt()
}

fn euclidean_distance_squared(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    (a - b).mapv(|x| x * x).sum()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_stochastic_manifold_learning_basic() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let sml = StochasticManifoldLearning::new(2)
            .batch_size(2)
            .n_epochs(10)
            .random_state(42);
        let fitted = sml.fit(&data, &()).unwrap();
        let embedding = fitted.transform(&data).unwrap();

        assert_eq!(embedding.shape(), &[4, 2]);
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_stochastic_manifold_learning_objectives() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let objectives = vec!["stress", "neighbor_preservation", "tsne"];

        for objective in objectives {
            let sml = StochasticManifoldLearning::new(2)
                .objective(objective)
                .batch_size(2)
                .n_epochs(5)
                .random_state(42);
            let fitted = sml.fit(&data, &()).unwrap();
            let embedding = fitted.transform(&data).unwrap();

            assert_eq!(embedding.shape(), &[4, 2]);
            assert!(embedding.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_stochastic_manifold_learning_loss_history() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let sml = StochasticManifoldLearning::new(2)
            .batch_size(2)
            .n_epochs(5)
            .random_state(42);
        let fitted = sml.fit(&data, &()).unwrap();

        assert_eq!(fitted.loss_history().len(), 5);
        assert!(fitted.loss_history().iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_stochastic_manifold_learning_partial_fit() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let new_data = array![[7.0, 8.0], [9.0, 10.0]];

        let sml = StochasticManifoldLearning::new(2)
            .batch_size(2)
            .random_state(42);
        let mut fitted = sml.fit(&data, &()).unwrap();

        let original_size = fitted.embedding().nrows();
        fitted.partial_fit(&new_data).unwrap();

        assert_eq!(fitted.embedding().nrows(), original_size + 2);
        assert!(fitted.embedding().iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_streaming_manifold_learning_basic() {
        let mut streaming = StreamingManifoldLearning::new(2, 10);

        let batch1 = array![[1.0, 2.0], [3.0, 4.0]];
        streaming.partial_fit(&batch1).unwrap();

        assert_eq!(streaming.buffer_size_current(), 2);

        let batch2 = array![[5.0, 6.0], [7.0, 8.0]];
        streaming.partial_fit(&batch2).unwrap();

        assert_eq!(streaming.buffer_size_current(), 4);

        let embedding = streaming.get_embedding();
        assert_eq!(embedding.shape(), &[4, 2]);
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_streaming_manifold_learning_buffer_overflow() {
        let mut streaming = StreamingManifoldLearning::new(2, 3);

        // Add more points than buffer size
        let batch = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

        streaming.partial_fit(&batch).unwrap();

        // Buffer should be limited to buffer_size
        assert_eq!(streaming.buffer_size_current(), 3);

        let embedding = streaming.get_embedding();
        assert_eq!(embedding.shape(), &[3, 2]);

        let data = streaming.get_data();
        assert_eq!(data.shape(), &[3, 2]);
    }

    #[test]
    fn test_streaming_manifold_learning_with_updates() {
        let mut streaming = StreamingManifoldLearning::new(2, 10)
            .update_frequency(2)
            .random_state(42);

        let batch1 = array![[1.0, 2.0]];
        streaming.partial_fit(&batch1).unwrap();

        let batch2 = array![[3.0, 4.0]];
        streaming.partial_fit(&batch2).unwrap();

        // Should trigger an update
        assert_eq!(streaming.buffer_size_current(), 2);

        let embedding = streaming.get_embedding();
        assert_eq!(embedding.shape(), &[2, 2]);
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_stochastic_manifold_learning_invalid_parameters() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];

        // Test n_components >= n_samples
        let sml = StochasticManifoldLearning::new(3);
        assert!(sml.fit(&data, &()).is_err());

        // Test batch_size > n_samples
        let sml = StochasticManifoldLearning::new(2).batch_size(5);
        assert!(sml.fit(&data, &()).is_err());
    }

    #[test]
    fn test_distance_functions() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let euclidean_dist = euclidean_distance(&a.view(), &b.view());
        assert_abs_diff_eq!(
            euclidean_dist,
            (3.0_f64.powi(2) * 3.0).sqrt(),
            epsilon = 1e-10
        );

        let euclidean_dist_sq = euclidean_distance_squared(&a.view(), &b.view());
        assert_abs_diff_eq!(euclidean_dist_sq, 3.0_f64.powi(2) * 3.0, epsilon = 1e-10);
    }
}
