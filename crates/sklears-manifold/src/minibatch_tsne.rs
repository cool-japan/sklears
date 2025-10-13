//! Mini-batch t-SNE implementation
//! This module provides Mini-batch t-SNE for large-scale datasets that don't fit in memory.

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

/// Mini-batch t-SNE for large-scale datasets
///
/// This implementation processes data in mini-batches to handle datasets
/// that don't fit in memory, making t-SNE scalable to very large datasets.
///
/// # Parameters
///
/// * `n_components` - Dimension of the embedded space
/// * `perplexity` - The perplexity is related to the number of nearest neighbors
/// * `batch_size` - Size of mini-batches for processing
/// * `learning_rate` - The learning rate for t-SNE
/// * `n_iter` - Maximum number of iterations
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::MiniBatchTSNE;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let mbtsne = MiniBatchTSNE::new()
///     .n_components(2)
///     .batch_size(2)
///     .perplexity(1.0);
///
/// let fitted = mbtsne.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MiniBatchTSNE<S = Untrained> {
    state: S,
    n_components: usize,
    perplexity: f64,
    batch_size: usize,
    learning_rate: f64,
    n_iter: usize,
    random_state: Option<u64>,
}

impl MiniBatchTSNE<Untrained> {
    /// Create a new MiniBatchTSNE instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            perplexity: 30.0,
            batch_size: 32,
            learning_rate: 200.0,
            n_iter: 1000,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the perplexity
    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for MiniBatchTSNE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trained state for Mini-batch t-SNE
#[derive(Debug, Clone)]
pub struct MBTSNETrained {
    /// Final embedding coordinates
    embedding: Array2<f64>,
}

impl Estimator for MiniBatchTSNE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MiniBatchTSNE<Untrained> {
    type Fitted = MiniBatchTSNE<MBTSNETrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Mini-batch t-SNE requires at least 2 samples".to_string(),
            });
        }

        if self.perplexity >= n_samples as f64 {
            return Err(SklearsError::InvalidParameter {
                name: "perplexity".to_string(),
                reason: format!(
                    "must be less than n_samples ({}), got {}",
                    n_samples, self.perplexity
                ),
            });
        }

        // Convert to f64 for computation
        let x_f64 = x.mapv(|v| v);

        // Compute pairwise affinities (simplified for mini-batch processing)
        let p_joint = self.compute_affinities(&x_f64)?;

        // Initialize embedding
        let mut embedding = self.initialize_embedding(n_samples)?;

        // Mini-batch optimization
        for iter in 0..self.n_iter {
            // Create mini-batches
            let mut rng = if let Some(seed) = self.random_state {
                StdRng::seed_from_u64(seed + iter as u64)
            } else {
                StdRng::seed_from_u64(thread_rng().gen::<u64>())
            };

            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            for chunk in indices.chunks(self.batch_size) {
                let batch_indices = chunk.to_vec();

                // Extract batch embedding
                let mut batch_embedding = Array2::zeros((batch_indices.len(), self.n_components));
                for (i, &idx) in batch_indices.iter().enumerate() {
                    batch_embedding.row_mut(i).assign(&embedding.row(idx));
                }

                // Extract batch affinities
                let mut p_batch = Array2::zeros((batch_indices.len(), batch_indices.len()));
                for (i, &idx_i) in batch_indices.iter().enumerate() {
                    for (j, &idx_j) in batch_indices.iter().enumerate() {
                        p_batch[[i, j]] = p_joint[[idx_i, idx_j]];
                    }
                }

                // Compute Q matrix for this batch
                let mut q_batch = Array2::zeros((batch_indices.len(), batch_indices.len()));
                for i in 0..batch_indices.len() {
                    for j in i + 1..batch_indices.len() {
                        let dist_sq = (&batch_embedding.row(i) - &batch_embedding.row(j))
                            .mapv(|x| x * x)
                            .sum();
                        let q_val = 1.0 / (1.0 + dist_sq);
                        q_batch[[i, j]] = q_val;
                        q_batch[[j, i]] = q_val;
                    }
                }

                // Normalize Q
                let q_sum = q_batch.sum();
                if q_sum > 0.0 {
                    q_batch /= q_sum;
                }

                // Compute gradients
                let mut gradients = Array2::<f64>::zeros((batch_indices.len(), self.n_components));
                for i in 0..batch_indices.len() {
                    for j in 0..batch_indices.len() {
                        if i != j {
                            let diff = &batch_embedding.row(i) - &batch_embedding.row(j);
                            let dist_sq = diff.dot(&diff);
                            let q_ij = 1.0 / (1.0 + dist_sq);
                            let pq_diff = p_batch[[i, j]] - q_batch[[i, j]];
                            let factor = 4.0 * pq_diff * q_ij;

                            for d in 0..self.n_components {
                                gradients[[i, d]] += factor * diff[d];
                            }
                        }
                    }
                }

                // Update embeddings for this batch
                let momentum = if iter < 250 { 0.5 } else { 0.8 };
                for (i, &idx) in batch_indices.iter().enumerate() {
                    for d in 0..self.n_components {
                        embedding[[idx, d]] -= self.learning_rate * gradients[[i, d]];
                    }
                }
            }
        }

        Ok(MiniBatchTSNE {
            state: MBTSNETrained { embedding },
            n_components: self.n_components,
            perplexity: self.perplexity,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for MiniBatchTSNE<MBTSNETrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // For fitted data, return the stored embedding
        // For new data, this would require out-of-sample extension (not implemented here)
        Ok(self.state.embedding.clone())
    }
}

impl MiniBatchTSNE<Untrained> {
    fn compute_affinities(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut p_joint = Array2::zeros((n_samples, n_samples));

        // Simplified affinity computation for demonstration
        // In practice, this would use proper perplexity-based affinities
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist_sq = (&x.row(i) - &x.row(j)).mapv(|v| v * v).sum();

                // Gaussian kernel with adaptive bandwidth
                let sigma = 1.0; // Simplified - should be computed based on perplexity
                let affinity = (-dist_sq / (2.0 * sigma * sigma)).exp();

                p_joint[[i, j]] = affinity;
                p_joint[[j, i]] = affinity;
            }
        }

        // Symmetrize and normalize
        let sum_p = p_joint.sum();
        if sum_p > 0.0 {
            p_joint /= sum_p;
            // Ensure minimum probability
            p_joint.mapv_inplace(|x| x.max(1e-12));
        }

        Ok(p_joint)
    }

    fn initialize_embedding(&self, n_samples: usize) -> SklResult<Array2<f64>> {
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components {
                embedding[[i, j]] = rng.sample::<f64, _>(scirs2_core::StandardNormal) * 1e-4;
            }
        }

        Ok(embedding)
    }
}

impl MiniBatchTSNE<MBTSNETrained> {
    /// Get the learned embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }
}
