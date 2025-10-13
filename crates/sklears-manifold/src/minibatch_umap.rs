//! Mini-batch UMAP implementation
//! This module provides Mini-batch UMAP for large-scale datasets that don't fit in memory.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Mini-batch UMAP for large-scale datasets
///
/// This implementation processes data in mini-batches to handle very large datasets
/// while maintaining the quality of UMAP embeddings.
///
/// # Parameters
///
/// * `n_components` - Dimension of the embedded space
/// * `n_neighbors` - Number of neighbors to consider
/// * `batch_size` - Size of mini-batches for processing
/// * `min_dist` - Minimum distance between embedded points
/// * `learning_rate` - Learning rate for optimization
/// * `n_epochs` - Number of training epochs
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::MiniBatchUMAP;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let mbumap = MiniBatchUMAP::new()
///     .n_components(2)
///     .batch_size(2)
///     .n_neighbors(2);
///
/// let fitted = mbumap.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MiniBatchUMAP<S = Untrained> {
    state: S,
    n_components: usize,
    n_neighbors: usize,
    batch_size: usize,
    min_dist: f64,
    learning_rate: f64,
    n_epochs: usize,
    random_state: Option<u64>,
}

impl MiniBatchUMAP<Untrained> {
    /// Create a new MiniBatchUMAP instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            n_neighbors: 15,
            batch_size: 32,
            min_dist: 0.1,
            learning_rate: 1.0,
            n_epochs: 200,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the minimum distance
    pub fn min_dist(mut self, min_dist: f64) -> Self {
        self.min_dist = min_dist;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of epochs
    pub fn n_epochs(mut self, n_epochs: usize) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for MiniBatchUMAP<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trained state for Mini-batch UMAP
#[derive(Debug, Clone)]
pub struct MBUMAPTrained {
    /// Final embedding coordinates
    embedding: Array2<f64>,
}

impl Estimator for MiniBatchUMAP<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MiniBatchUMAP<Untrained> {
    type Fitted = MiniBatchUMAP<MBUMAPTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Mini-batch UMAP requires at least 2 samples".to_string(),
            });
        }

        if self.n_neighbors >= n_samples {
            return Err(SklearsError::InvalidParameter {
                name: "n_neighbors".to_string(),
                reason: format!(
                    "must be less than n_samples ({}), got {}",
                    n_samples, self.n_neighbors
                ),
            });
        }

        // Convert to f64 for computation
        let x_f64 = x.mapv(|v| v);

        // Build k-NN graph (simplified for mini-batch)
        let knn_graph = self.build_knn_graph(&x_f64)?;

        // Initialize embedding
        let mut embedding = self.initialize_embedding(n_samples)?;

        // Mini-batch optimization loop
        for epoch in 0..self.n_epochs {
            let mut rng = if let Some(seed) = self.random_state {
                StdRng::seed_from_u64(seed + epoch as u64)
            } else {
                StdRng::seed_from_u64(thread_rng().gen::<u64>())
            };

            // Create mini-batches
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            for chunk in indices.chunks(self.batch_size) {
                let batch_indices = chunk.to_vec();

                // Process mini-batch
                self.process_minibatch(&knn_graph, &mut embedding, &batch_indices, epoch)?;
            }
        }

        Ok(MiniBatchUMAP {
            state: MBUMAPTrained { embedding },
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            batch_size: self.batch_size,
            min_dist: self.min_dist,
            learning_rate: self.learning_rate,
            n_epochs: self.n_epochs,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for MiniBatchUMAP<MBUMAPTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // For fitted data, return the stored embedding
        // For new data, this would require out-of-sample extension (not implemented here)
        Ok(self.state.embedding.clone())
    }
}

impl MiniBatchUMAP<Untrained> {
    fn build_knn_graph(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut knn_graph = Array2::zeros((n_samples, n_samples));

        // Simple k-NN graph construction
        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = (&x.row(i) - &x.row(j)).mapv(|v| v * v).sum().sqrt();
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Connect to k nearest neighbors
            for &(j, dist) in distances.iter().take(self.n_neighbors) {
                let weight = (-dist).exp(); // Gaussian weight
                knn_graph[(i, j)] = weight;
            }
        }

        // Symmetrize
        for i in 0..n_samples {
            for j in 0..n_samples {
                knn_graph[(i, j)] = knn_graph[(i, j)].max(knn_graph[(j, i)]);
                knn_graph[(j, i)] = knn_graph[(i, j)];
            }
        }

        Ok(knn_graph)
    }

    fn initialize_embedding(&self, n_samples: usize) -> SklResult<Array2<f64>> {
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut embedding = Array2::zeros((n_samples, self.n_components));
        let scale = 10.0;

        for i in 0..n_samples {
            for j in 0..self.n_components {
                embedding[[i, j]] = rng.sample::<f64, _>(scirs2_core::StandardNormal) * scale;
            }
        }

        Ok(embedding)
    }

    fn process_minibatch(
        &self,
        knn_graph: &Array2<f64>,
        embedding: &mut Array2<f64>,
        batch_indices: &[usize],
        epoch: usize,
    ) -> SklResult<()> {
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed + epoch as u64)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        // Process positive samples (attractive forces)
        for &i in batch_indices {
            for j in 0..knn_graph.ncols() {
                let weight = knn_graph[(i, j)];
                if weight > 0.0 && i != j {
                    self.apply_attractive_force(embedding, i, j, weight);
                }
            }

            // Process negative samples (repulsive forces)
            for _ in 0..5 {
                // Sample negative examples
                let neg_j = rng.gen_range(0..embedding.nrows());
                if neg_j != i {
                    self.apply_repulsive_force(embedding, i, neg_j);
                }
            }
        }

        Ok(())
    }

    fn apply_attractive_force(&self, embedding: &mut Array2<f64>, i: usize, j: usize, weight: f64) {
        let dist_sq = (&embedding.row(i) - &embedding.row(j))
            .mapv(|x| x * x)
            .sum();

        let grad_coeff = -2.0 * weight * self.learning_rate / (1.0 + dist_sq);

        for d in 0..self.n_components {
            let diff = embedding[[i, d]] - embedding[[j, d]];
            let update = grad_coeff * diff;

            embedding[[i, d]] += update;
            embedding[[j, d]] -= update;
        }
    }

    fn apply_repulsive_force(&self, embedding: &mut Array2<f64>, i: usize, j: usize) {
        let dist_sq = (&embedding.row(i) - &embedding.row(j))
            .mapv(|x| x * x)
            .sum();

        if dist_sq > 0.0 {
            let grad_coeff = 2.0 * self.learning_rate
                / ((self.min_dist * self.min_dist + dist_sq) * (1.0 + dist_sq));

            for d in 0..self.n_components {
                let diff = embedding[[i, d]] - embedding[[j, d]];
                let update = grad_coeff * diff;

                embedding[[i, d]] += update;
                embedding[[j, d]] -= update;
            }
        }
    }
}

impl MiniBatchUMAP<MBUMAPTrained> {
    /// Get the learned embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }
}
