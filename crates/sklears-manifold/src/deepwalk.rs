//! DeepWalk Algorithm implementation
//! This module provides DeepWalk for learning continuous representations of vertices in graphs.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// DeepWalk Algorithm
///
/// DeepWalk learns continuous representations of vertices in a graph by modeling
/// a stream of short random walks. It treats random walk sequences like sentences
/// and uses skip-gram to learn representations.
#[derive(Debug, Clone)]
pub struct DeepWalk<S = Untrained> {
    state: S,
    n_components: usize,
    walk_length: usize,
    num_walks: usize,
    window_size: usize,
    min_count: usize,
    epochs: usize,
    learning_rate: f64,
    random_state: Option<u64>,
}

impl DeepWalk<Untrained> {
    /// Create a new DeepWalk instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 128,
            walk_length: 40,
            num_walks: 80,
            window_size: 5,
            min_count: 0,
            epochs: 1,
            learning_rate: 0.025,
            random_state: None,
        }
    }

    /// Set embedding dimensions
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set walk length
    pub fn walk_length(mut self, walk_length: usize) -> Self {
        self.walk_length = walk_length;
        self
    }

    /// Set number of walks per node
    pub fn num_walks(mut self, num_walks: usize) -> Self {
        self.num_walks = num_walks;
        self
    }

    /// Set context window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set minimum frequency count
    pub fn min_count(mut self, min_count: usize) -> Self {
        self.min_count = min_count;
        self
    }

    /// Set number of training epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for DeepWalk<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trained state for DeepWalk
#[derive(Debug, Clone)]
pub struct DeepWalkTrained {
    embeddings: Array2<f64>,
    vocabulary: HashMap<usize, usize>,
}

impl Estimator for DeepWalk<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for DeepWalk<DeepWalkTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for DeepWalk<Untrained> {
    type Fitted = DeepWalk<DeepWalkTrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        // Build adjacency matrix from input data
        let adjacency = self.build_adjacency_matrix(x)?;

        // Generate random walks
        let walks = self.generate_random_walks(&adjacency)?;

        // Build vocabulary from walks
        let vocabulary = self.build_vocabulary(&walks);

        // Train skip-gram model
        let embeddings = self.train_skip_gram(&walks, &vocabulary)?;

        Ok(DeepWalk {
            state: DeepWalkTrained {
                embeddings,
                vocabulary,
            },
            n_components: self.n_components,
            walk_length: self.walk_length,
            num_walks: self.num_walks,
            window_size: self.window_size,
            min_count: self.min_count,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for DeepWalk<DeepWalkTrained> {
    fn transform(&self, _x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        // Return the learned embeddings
        Ok(self.state.embeddings.clone())
    }
}

impl DeepWalk<Untrained> {
    fn build_adjacency_matrix(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut adjacency = Array2::zeros((n_samples, n_samples));

        // Simple k-NN graph construction
        let k = 5.min(n_samples - 1);

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

    fn generate_random_walks(&self, adjacency: &Array2<f64>) -> SklResult<Vec<Vec<usize>>> {
        let n_nodes = adjacency.nrows();
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        let mut all_walks = Vec::new();

        for start_node in 0..n_nodes {
            for _ in 0..self.num_walks {
                let mut walk = vec![start_node];
                let mut current_node = start_node;

                for _ in 1..self.walk_length {
                    // Find neighbors
                    let neighbors: Vec<usize> = (0..n_nodes)
                        .filter(|&j| adjacency[(current_node, j)] > 0.0)
                        .collect();

                    if neighbors.is_empty() {
                        break;
                    }

                    // Uniform random choice
                    let next_node = neighbors[rng.gen_range(0..neighbors.len())];
                    walk.push(next_node);
                    current_node = next_node;
                }

                if walk.len() > 1 {
                    all_walks.push(walk);
                }
            }
        }

        Ok(all_walks)
    }

    fn build_vocabulary(&self, walks: &[Vec<usize>]) -> HashMap<usize, usize> {
        let mut node_counts: HashMap<usize, usize> = HashMap::new();

        // Count occurrences of each node
        for walk in walks {
            for &node in walk {
                *node_counts.entry(node).or_insert(0) += 1;
            }
        }

        // Build vocabulary with nodes that meet minimum count requirement
        let mut vocabulary = HashMap::new();
        let mut vocab_id = 0;

        for (node, count) in node_counts {
            if count >= self.min_count {
                vocabulary.insert(node, vocab_id);
                vocab_id += 1;
            }
        }

        vocabulary
    }

    fn train_skip_gram(
        &self,
        walks: &[Vec<usize>],
        vocabulary: &std::collections::HashMap<usize, usize>,
    ) -> SklResult<Array2<f64>> {
        let vocab_size = vocabulary.len();
        let mut embeddings = Array2::zeros((vocab_size, self.n_components));
        let mut rng = thread_rng();

        // Fill with random values
        for mut row in embeddings.rows_mut() {
            for elem in row.iter_mut() {
                *elem = rng.sample(scirs2_core::StandardNormal);
            }
        }

        // Normalize embeddings
        for mut row in embeddings.rows_mut() {
            let norm = row.mapv(|x: f64| x * x).sum().sqrt();
            if norm > 1e-10 {
                row /= norm;
            }
        }

        // Training loop (simplified - in practice would use hierarchical softmax)
        for _ in 0..self.epochs {
            for walk in walks {
                for (i, &center_node) in walk.iter().enumerate() {
                    if let Some(&center_id) = vocabulary.get(&center_node) {
                        // Define context window
                        let start = i.saturating_sub(self.window_size);
                        let end = (i + self.window_size + 1).min(walk.len());

                        for j in start..end {
                            if i != j {
                                if let Some(&context_node) = walk.get(j) {
                                    if let Some(&context_id) = vocabulary.get(&context_node) {
                                        // Simple gradient update (simplified skip-gram)
                                        let dot_product: f64 = embeddings
                                            .row(center_id)
                                            .iter()
                                            .zip(embeddings.row(context_id).iter())
                                            .map(|(a, b)| a * b)
                                            .sum();

                                        let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                                        let gradient = self.learning_rate * (1.0 - sigmoid);

                                        for k in 0..self.n_components {
                                            let center_val = embeddings[(center_id, k)];
                                            let context_val = embeddings[(context_id, k)];

                                            embeddings[(center_id, k)] += gradient * context_val;
                                            embeddings[(context_id, k)] += gradient * center_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(embeddings)
    }
}

impl DeepWalk<DeepWalkTrained> {
    /// Get the learned embeddings
    pub fn embeddings(&self) -> &Array2<f64> {
        &self.state.embeddings
    }

    /// Get the vocabulary mapping
    pub fn vocabulary(&self) -> &HashMap<usize, usize> {
        &self.state.vocabulary
    }
}
