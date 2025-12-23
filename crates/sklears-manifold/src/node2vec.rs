//! Node2Vec Algorithm implementation
//!
//! This module provides Node2Vec for learning continuous feature representations for nodes in networks.

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

/// Node2Vec Algorithm
///
/// Node2vec is a framework for learning continuous feature representations for nodes
/// in networks. It uses biased random walks to generate contexts that preserve both
/// local and global network structure.
#[derive(Debug, Clone)]
pub struct Node2Vec<S = Untrained> {
    state: S,
    n_components: usize,
    walk_length: usize,
    num_walks: usize,
    p: f64, // Return parameter
    q: f64, // In-out parameter
    window_size: usize,
    min_count: usize,
    batch_words: usize,
    epochs: usize,
    learning_rate: f64,
    negative_samples: usize,
    random_state: Option<u64>,
}

impl Node2Vec<Untrained> {
    /// Create a new Node2Vec instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 128,
            walk_length: 80,
            num_walks: 10,
            p: 1.0,
            q: 1.0,
            window_size: 10,
            min_count: 1,
            batch_words: 4,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 5,
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

    /// Set the number of walks
    pub fn num_walks(mut self, num_walks: usize) -> Self {
        self.num_walks = num_walks;
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

    /// Set the window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the minimum count
    pub fn min_count(mut self, min_count: usize) -> Self {
        self.min_count = min_count;
        self
    }

    /// Set the batch words
    pub fn batch_words(mut self, batch_words: usize) -> Self {
        self.batch_words = batch_words;
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

    /// Set the number of negative samples
    pub fn negative_samples(mut self, negative_samples: usize) -> Self {
        self.negative_samples = negative_samples;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for Node2Vec<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trained state for Node2Vec
#[derive(Debug, Clone)]
pub struct Node2VecTrained {
    /// Node embeddings
    node_embeddings: Array2<f64>,
    /// Vocabulary mapping
    vocab: HashMap<usize, usize>,
}

impl Estimator for Node2Vec<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for Node2Vec<Node2VecTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for Node2Vec<Untrained> {
    type Fitted = Node2Vec<Node2VecTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Node2Vec requires at least 2 samples".to_string(),
            });
        }

        // Convert to f64 for computation
        let x_f64 = x.mapv(|v| v);

        // Build adjacency matrix from data
        let adjacency = self.build_adjacency_matrix(&x_f64)?;

        // Generate biased random walks using Node2Vec parameters
        let walks = self.generate_node2vec_walks(&adjacency)?;

        // Train skip-gram model on walks
        let (node_embeddings, vocab) = self.train_skipgram_on_walks(&walks)?;

        Ok(Node2Vec {
            state: Node2VecTrained {
                node_embeddings,
                vocab,
            },
            n_components: self.n_components,
            walk_length: self.walk_length,
            num_walks: self.num_walks,
            p: self.p,
            q: self.q,
            window_size: self.window_size,
            min_count: self.min_count,
            batch_words: self.batch_words,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            negative_samples: self.negative_samples,
            random_state: self.random_state,
        })
    }
}

impl Node2Vec<Untrained> {
    fn build_adjacency_matrix(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
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
            for &(j, dist) in distances.iter().take(k) {
                let weight = (-dist).exp(); // Gaussian weight
                adjacency[(i, j)] = weight;
                adjacency[(j, i)] = weight; // Make symmetric
            }
        }

        Ok(adjacency)
    }

    fn generate_node2vec_walks(&self, adjacency: &Array2<f64>) -> SklResult<Vec<Vec<usize>>> {
        let n_nodes = adjacency.nrows();
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        let mut all_walks = Vec::new();

        for start_node in 0..n_nodes {
            for _ in 0..self.num_walks {
                let walk = self.node2vec_walk(start_node, adjacency, &mut rng)?;
                if walk.len() >= 2 {
                    all_walks.push(walk);
                }
            }
        }

        Ok(all_walks)
    }

    fn node2vec_walk(
        &self,
        start_node: usize,
        adjacency: &Array2<f64>,
        rng: &mut StdRng,
    ) -> SklResult<Vec<usize>> {
        let mut walk = vec![start_node];
        let mut prev_node = None;
        let mut current_node = start_node;

        for _ in 1..self.walk_length {
            let neighbors = self.get_neighbors(current_node, adjacency);

            if neighbors.is_empty() {
                break;
            }

            let next_node = if let Some(prev) = prev_node {
                self.biased_choice(current_node, prev, &neighbors, adjacency, rng)?
            } else {
                // First step: uniform random choice
                neighbors[rng.gen_range(0..neighbors.len())]
            };

            walk.push(next_node);
            prev_node = Some(current_node);
            current_node = next_node;
        }

        Ok(walk)
    }

    fn get_neighbors(&self, node: usize, adjacency: &Array2<f64>) -> Vec<usize> {
        adjacency
            .row(node)
            .iter()
            .enumerate()
            .filter_map(|(idx, &weight)| if weight > 0.0 { Some(idx) } else { None })
            .collect()
    }

    fn biased_choice(
        &self,
        current: usize,
        prev: usize,
        neighbors: &[usize],
        adjacency: &Array2<f64>,
        rng: &mut StdRng,
    ) -> SklResult<usize> {
        let mut weights = Vec::new();
        let mut total_weight = 0.0;

        for &neighbor in neighbors {
            let edge_weight = adjacency[(current, neighbor)];

            let bias = if neighbor == prev {
                // Return to previous node
                1.0 / self.p
            } else if adjacency[(prev, neighbor)] > 0.0 {
                // Neighbor is also connected to previous node (local exploration)
                1.0
            } else {
                // Move away from previous node (global exploration)
                1.0 / self.q
            };

            let final_weight = edge_weight * bias;
            weights.push(final_weight);
            total_weight += final_weight;
        }

        if total_weight <= 0.0 {
            // Fallback to uniform choice
            return Ok(neighbors[rng.gen_range(0..neighbors.len())]);
        }

        // Weighted random choice
        let mut cumulative = 0.0;
        let threshold = rng.gen::<f64>() * total_weight;

        for (i, &weight) in weights.iter().enumerate() {
            cumulative += weight;
            if cumulative >= threshold {
                return Ok(neighbors[i]);
            }
        }

        // Fallback (should not reach here)
        Ok(neighbors[neighbors.len() - 1])
    }

    fn train_skipgram_on_walks(
        &self,
        walks: &[Vec<usize>],
    ) -> SklResult<(Array2<f64>, HashMap<usize, usize>)> {
        // Build vocabulary
        let mut word_count = HashMap::new();
        for walk in walks {
            for &word in walk {
                *word_count.entry(word).or_insert(0) += 1;
            }
        }

        // Filter by min_count
        let vocab: HashMap<usize, usize> = word_count
            .iter()
            .filter(|(_, &count)| count >= self.min_count)
            .enumerate()
            .map(|(idx, (&word, _))| (word, idx))
            .collect();

        let vocab_size = vocab.len();
        if vocab_size == 0 {
            return Err(SklearsError::InvalidInput(
                "No words meet minimum count requirement".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        // Initialize embeddings
        let mut node_embeddings = Array2::zeros((vocab_size, self.n_components));
        for i in 0..vocab_size {
            for j in 0..self.n_components {
                node_embeddings[(i, j)] = rng.sample::<f64, _>(scirs2_core::StandardNormal) * 0.1;
            }
        }

        // Simplified skip-gram training
        for _epoch in 0..self.epochs {
            for walk in walks {
                for (center_idx, &center_word) in walk.iter().enumerate() {
                    if let Some(&center_vocab_idx) = vocab.get(&center_word) {
                        // Context window
                        let start = center_idx.saturating_sub(self.window_size);
                        let end = (center_idx + self.window_size + 1).min(walk.len());

                        for context_idx in start..end {
                            if context_idx != center_idx {
                                if let Some(&context_word) = walk.get(context_idx) {
                                    if let Some(&context_vocab_idx) = vocab.get(&context_word) {
                                        // Simplified gradient update
                                        let dot_product: f64 = node_embeddings
                                            .row(center_vocab_idx)
                                            .iter()
                                            .zip(node_embeddings.row(context_vocab_idx).iter())
                                            .map(|(a, b)| a * b)
                                            .sum();

                                        let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                                        let gradient = self.learning_rate * (1.0 - sigmoid);

                                        for k in 0..self.n_components {
                                            let center_val = node_embeddings[(center_vocab_idx, k)];
                                            let context_val =
                                                node_embeddings[(context_vocab_idx, k)];

                                            node_embeddings[(center_vocab_idx, k)] +=
                                                gradient * context_val;
                                            node_embeddings[(context_vocab_idx, k)] +=
                                                gradient * center_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok((node_embeddings, vocab))
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for Node2Vec<Node2VecTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, _) = x.dim();

        if n_samples != self.state.vocab.len() {
            return Err(SklearsError::InvalidInput(
                "Input size must match training data size for Node2Vec".to_string(),
            ));
        }

        // Return the learned embeddings
        Ok(self.state.node_embeddings.mapv(|v| v as Float))
    }
}

impl Node2Vec<Node2VecTrained> {
    /// Get the learned node embeddings
    pub fn node_embeddings(&self) -> &Array2<f64> {
        &self.state.node_embeddings
    }

    /// Get the vocabulary mapping
    pub fn vocab(&self) -> &HashMap<usize, usize> {
        &self.state.vocab
    }
}
