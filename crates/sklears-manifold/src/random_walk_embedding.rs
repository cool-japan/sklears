//! Random Walk Embedding implementation
//!
//! This module provides Random Walk Embeddings for manifold learning through random walk sampling on graphs.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Random Walk Embeddings
///
/// Random walk embeddings learn representations by modeling the probability
/// distributions of random walks on the data graph. This includes methods like
/// DeepWalk and node2vec that use skip-gram models to learn embeddings.
#[derive(Debug, Clone)]
pub struct RandomWalkEmbedding<S = Untrained> {
    state: S,
    n_components: usize,
    walk_length: usize,
    num_walks: usize,
    window_size: usize,
    p: f64, // Return parameter for node2vec
    q: f64, // In-out parameter for node2vec
    workers: usize,
    epochs: usize,
    learning_rate: f64,
    random_state: Option<u64>,
}

impl RandomWalkEmbedding<Untrained> {
    /// Create a new RandomWalkEmbedding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 128,
            walk_length: 80,
            num_walks: 10,
            window_size: 10,
            p: 1.0,
            q: 1.0,
            workers: 1,
            epochs: 1,
            learning_rate: 0.025,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the walk length
    pub fn walk_length(mut self, walk_length: usize) -> Self {
        self.walk_length = walk_length;
        self
    }

    /// Set the number of walks per node
    pub fn num_walks(mut self, num_walks: usize) -> Self {
        self.num_walks = num_walks;
        self
    }

    /// Set the window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the return parameter (p)
    pub fn p(mut self, p: f64) -> Self {
        self.p = p;
        self
    }

    /// Set the in-out parameter (q)
    pub fn q(mut self, q: f64) -> Self {
        self.q = q;
        self
    }

    /// Set the number of workers
    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }

    /// Set the number of epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for RandomWalkEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trained state for Random Walk Embedding
#[derive(Debug, Clone)]
pub struct RandomWalkEmbeddingTrained {
    /// Node embeddings
    embedding: Array2<f64>,
    /// Vocabulary mapping from node indices to embedding indices
    vocab: HashMap<usize, usize>,
}

impl Estimator for RandomWalkEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for RandomWalkEmbedding<RandomWalkEmbeddingTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for RandomWalkEmbedding<Untrained> {
    type Fitted = RandomWalkEmbedding<RandomWalkEmbeddingTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Random Walk Embedding requires at least 2 samples".to_string(),
            });
        }

        // Convert to f64 for computation
        let x_f64 = x.mapv(|v| v);

        // Build adjacency matrix/graph from data
        let adjacency = self.build_graph(&x_f64)?;

        // Generate random walks
        let walks = self.generate_walks(&adjacency)?;

        // Train skip-gram model to learn embeddings
        let (embedding, vocab) = self.train_skipgram(&walks)?;

        Ok(RandomWalkEmbedding {
            state: RandomWalkEmbeddingTrained { embedding, vocab },
            n_components: self.n_components,
            walk_length: self.walk_length,
            num_walks: self.num_walks,
            window_size: self.window_size,
            p: self.p,
            q: self.q,
            workers: self.workers,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            random_state: self.random_state,
        })
    }
}

impl RandomWalkEmbedding<Untrained> {
    fn build_graph(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut adjacency = Array2::zeros((n_samples, n_samples));

        // Simple k-NN graph construction
        let k = 10.min(n_samples - 1);

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
            for &(j, _) in distances.iter().take(k) {
                adjacency[(i, j)] = 1.0;
                adjacency[(j, i)] = 1.0; // Make symmetric
            }
        }

        Ok(adjacency)
    }

    fn generate_walks(&self, adjacency: &Array2<f64>) -> SklResult<Vec<Vec<usize>>> {
        let n_nodes = adjacency.nrows();
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut all_walks = Vec::new();

        for node in 0..n_nodes {
            for _ in 0..self.num_walks {
                let walk = self.random_walk(node, adjacency, &mut rng)?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    fn random_walk(
        &self,
        start_node: usize,
        adjacency: &Array2<f64>,
        rng: &mut StdRng,
    ) -> SklResult<Vec<usize>> {
        let mut walk = vec![start_node];
        let mut current = start_node;

        for _ in 1..self.walk_length {
            // Find neighbors
            let neighbors: Vec<usize> = (0..adjacency.ncols())
                .filter(|&j| adjacency[(current, j)] > 0.0)
                .collect();

            if neighbors.is_empty() {
                break;
            }

            // Simple uniform random choice for now
            // TODO: Implement biased sampling for node2vec (p, q parameters)
            let next_idx = rng.gen_range(0..neighbors.len());
            current = neighbors[next_idx];
            walk.push(current);
        }

        Ok(walk)
    }

    fn train_skipgram(
        &self,
        walks: &[Vec<usize>],
    ) -> SklResult<(Array2<f64>, HashMap<usize, usize>)> {
        // Build vocabulary
        let mut vocab = HashMap::new();
        let mut vocab_count = 0;

        for walk in walks {
            for &node in walk {
                vocab.entry(node).or_insert_with(|| {
                    vocab_count += 1;
                    vocab_count - 1
                });
            }
        }

        let vocab_size = vocab.len();
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        // Initialize embeddings randomly
        let mut embedding = Array2::zeros((vocab_size, self.n_components));
        for i in 0..vocab_size {
            for j in 0..self.n_components {
                embedding[(i, j)] = rng.sample::<f64, _>(scirs2_core::StandardNormal) * 0.1;
            }
        }

        // Simplified skip-gram training (in practice would use hierarchical softmax/negative sampling)
        for _epoch in 0..self.epochs {
            for walk in walks {
                for (i, &center_word) in walk.iter().enumerate() {
                    let center_idx = vocab[&center_word];

                    for j in (i.saturating_sub(self.window_size))
                        ..(i + self.window_size + 1).min(walk.len())
                    {
                        if i != j {
                            let context_word = walk[j];
                            let context_idx = vocab[&context_word];

                            // Simple gradient update (simplified)
                            let dot_product: f64 = embedding
                                .row(center_idx)
                                .iter()
                                .zip(embedding.row(context_idx).iter())
                                .map(|(a, b)| a * b)
                                .sum();

                            let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                            let gradient = self.learning_rate * (1.0 - sigmoid);

                            for k in 0..self.n_components {
                                let center_val = embedding[(center_idx, k)];
                                let context_val = embedding[(context_idx, k)];

                                embedding[(center_idx, k)] += gradient * context_val;
                                embedding[(context_idx, k)] += gradient * center_val;
                            }
                        }
                    }
                }
            }
        }

        // Normalize embeddings
        for mut row in embedding.rows_mut() {
            let norm = row.mapv(|x: f64| x * x).sum().sqrt();
            if norm > 1e-10 {
                row /= norm;
            }
        }

        Ok((embedding, vocab))
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for RandomWalkEmbedding<RandomWalkEmbeddingTrained>
{
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, _) = x.dim();

        if n_samples != self.state.vocab.len() {
            return Err(SklearsError::InvalidInput(
                "Input size must match training data size for Random Walk Embedding".to_string(),
            ));
        }

        // Return the learned embeddings
        Ok(self.state.embedding.mapv(|v| v as Float))
    }
}

impl RandomWalkEmbedding<RandomWalkEmbeddingTrained> {
    /// Get the learned embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the vocabulary mapping
    pub fn vocab(&self) -> &HashMap<usize, usize> {
        &self.state.vocab
    }
}
