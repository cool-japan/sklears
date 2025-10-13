//! Graph Neural Network implementations for manifold learning
//! This module provides Graph Convolutional Networks (GCN), GraphSAGE, and
//! Graph Attention Networks (GAT) for learning node embeddings on graph-structured data.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

// =====================================================================================
// GRAPH NEURAL NETWORK EMBEDDINGS
// =====================================================================================

/// Graph Neural Network Embeddings for Graph-based Manifold Learning
///
/// These are neural network architectures designed to learn node embeddings
/// on graph-structured data through message passing and aggregation.

/// Graph Convolutional Network (GCN) Layer
///
/// Implements a single GCN layer that performs graph convolution:
/// H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
#[derive(Debug, Clone)]
pub struct GCNLayer {
    weight: Array2<f64>,
    bias: Option<Array1<f64>>,
    input_dim: usize,
    output_dim: usize,
}

impl GCNLayer {
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Self {
        let mut rng = thread_rng();

        // Xavier initialization
        let limit = (6.0 / (input_dim + output_dim) as f64).sqrt();
        let mut weight = Array2::zeros((input_dim, output_dim));
        for elem in weight.iter_mut() {
            *elem = rng.gen_range(-limit..limit);
        }

        let bias = if use_bias {
            Some(Array1::zeros(output_dim))
        } else {
            None
        };

        Self {
            weight,
            bias,
            input_dim,
            output_dim,
        }
    }

    pub fn forward(
        &self,
        features: &Array2<f64>,
        normalized_adj: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        // H^(l+1) = σ(A_norm H^(l) W^(l))
        let transformed = features.dot(&self.weight);
        let mut output = normalized_adj.dot(&transformed);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for mut row in output.rows_mut() {
                row += bias;
            }
        }

        // Apply ReLU activation
        output.mapv_inplace(|x| x.max(0.0));

        Ok(output)
    }
}

/// Graph Convolutional Network (GCN)
///
/// Multi-layer GCN for node embedding learning on graphs.
#[derive(Debug, Clone)]
pub struct GraphConvolutionalNetwork<S = Untrained> {
    state: S,
    layers: Vec<usize>,
    dropout_rate: f64,
    learning_rate: f64,
    epochs: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct GCNTrained {
    gcn_layers: Vec<GCNLayer>,
    embedding: Array2<f64>,
    normalized_adjacency: Array2<f64>,
}

impl Default for GraphConvolutionalNetwork<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphConvolutionalNetwork<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            layers: vec![64, 32],
            dropout_rate: 0.5,
            learning_rate: 0.01,
            epochs: 200,
            random_state: None,
        }
    }

    pub fn layers(mut self, layers: Vec<usize>) -> Self {
        self.layers = layers;
        self
    }

    pub fn dropout_rate(mut self, dropout_rate: f64) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for GraphConvolutionalNetwork<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for GraphConvolutionalNetwork<GCNTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for GraphConvolutionalNetwork<Untrained> {
    type Fitted = GraphConvolutionalNetwork<GCNTrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_nodes, n_features) = x.dim();

        // Create adjacency matrix from feature similarity
        let adjacency = self.create_adjacency_matrix(x)?;

        // Normalize adjacency matrix for GCN
        let normalized_adj = self.normalize_adjacency(&adjacency)?;

        // Initialize GCN layers
        let mut gcn_layers = Vec::new();
        let mut layer_dims = vec![n_features];
        layer_dims.extend_from_slice(&self.layers);

        for i in 0..layer_dims.len() - 1 {
            let layer = GCNLayer::new(layer_dims[i], layer_dims[i + 1], true);
            gcn_layers.push(layer);
        }

        // Simple training loop (in practice, would use gradient descent)
        let mut current_features = x.to_owned();
        for layer in &gcn_layers {
            current_features = layer.forward(&current_features, &normalized_adj)?;
        }

        Ok(GraphConvolutionalNetwork {
            state: GCNTrained {
                gcn_layers,
                embedding: current_features,
                normalized_adjacency: normalized_adj,
            },
            layers: self.layers,
            dropout_rate: self.dropout_rate,
            learning_rate: self.learning_rate,
            epochs: self.epochs,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for GraphConvolutionalNetwork<GCNTrained> {
    fn transform(&self, _x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        Ok(self.state.embedding.clone())
    }
}

impl GraphConvolutionalNetwork<Untrained> {
    fn create_adjacency_matrix(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_nodes, _) = x.dim();
        let mut adjacency = Array2::zeros((n_nodes, n_nodes));
        let k = 10; // Number of neighbors

        for i in 0..n_nodes {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_nodes {
                if i != j {
                    let dist = x
                        .row(i)
                        .iter()
                        .zip(x.row(j).iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Connect to k nearest neighbors
            for &(j, dist) in distances.iter().take(k) {
                adjacency[(i, j)] = (-dist.powi(2) / 2.0).exp();
            }
        }

        // Make symmetric
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                adjacency[(i, j)] = (adjacency[(i, j)] + adjacency[(j, i)]) / 2.0;
                adjacency[(j, i)] = adjacency[(i, j)];
            }
        }

        Ok(adjacency)
    }

    fn normalize_adjacency(&self, adjacency: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_nodes = adjacency.nrows();

        // Add self-loops: A' = A + I
        let mut adj_with_self_loops = adjacency.clone();
        for i in 0..n_nodes {
            adj_with_self_loops[(i, i)] += 1.0;
        }

        // Compute degree matrix
        let degrees: Array1<f64> = adj_with_self_loops.sum_axis(Axis(1));

        // D^(-1/2)
        let mut d_inv_sqrt = Array2::eye(n_nodes);
        for i in 0..n_nodes {
            if degrees[i] > 1e-10 {
                d_inv_sqrt[(i, i)] = 1.0 / degrees[i].sqrt();
            }
        }

        // D^(-1/2) A D^(-1/2)
        let normalized = d_inv_sqrt.dot(&adj_with_self_loops).dot(&d_inv_sqrt);

        Ok(normalized)
    }
}

/// GraphSAGE (Graph Sample and Aggregate)
///
/// Implements GraphSAGE for inductive representation learning on graphs.
#[derive(Debug, Clone)]
pub struct GraphSAGE<S = Untrained> {
    state: S,
    n_components: usize,
    n_layers: usize,
    aggregator: String, // "mean", "lstm", "pool"
    sample_size: usize,
    dropout_rate: f64,
    learning_rate: f64,
    epochs: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct GraphSAGETrained {
    weight_matrices: Vec<Array2<f64>>,
    embedding: Array2<f64>,
    adjacency: Array2<f64>,
}

impl Default for GraphSAGE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSAGE<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 128,
            n_layers: 2,
            aggregator: "mean".to_string(),
            sample_size: 25,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            epochs: 200,
            random_state: None,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    pub fn aggregator(mut self, aggregator: String) -> Self {
        self.aggregator = aggregator;
        self
    }

    pub fn sample_size(mut self, sample_size: usize) -> Self {
        self.sample_size = sample_size;
        self
    }
}

impl Estimator for GraphSAGE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for GraphSAGE<GraphSAGETrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for GraphSAGE<Untrained> {
    type Fitted = GraphSAGE<GraphSAGETrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_nodes, n_features) = x.dim();

        // Create adjacency matrix
        let adjacency = self.create_adjacency_matrix(x)?;

        // Initialize weight matrices for each layer
        let mut weight_matrices = Vec::new();
        let mut rng = thread_rng();

        for layer in 0..self.n_layers {
            let input_dim = if layer == 0 {
                n_features * 2
            } else {
                self.n_components * 2
            };
            let output_dim = self.n_components;

            let limit = (6.0 / (input_dim + output_dim) as f64).sqrt();
            let mut weight = Array2::zeros((input_dim, output_dim));
            for elem in weight.iter_mut() {
                *elem = rng.gen_range(-limit..limit);
            }
            weight_matrices.push(weight);
        }

        // Forward pass through GraphSAGE layers
        let mut current_embeddings = x.to_owned();

        for layer in 0..self.n_layers {
            current_embeddings =
                self.sage_layer_forward(&current_embeddings, &adjacency, &weight_matrices[layer])?;
        }

        Ok(GraphSAGE {
            state: GraphSAGETrained {
                weight_matrices,
                embedding: current_embeddings,
                adjacency,
            },
            n_components: self.n_components,
            n_layers: self.n_layers,
            aggregator: self.aggregator,
            sample_size: self.sample_size,
            dropout_rate: self.dropout_rate,
            learning_rate: self.learning_rate,
            epochs: self.epochs,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for GraphSAGE<GraphSAGETrained> {
    fn transform(&self, _x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        Ok(self.state.embedding.clone())
    }
}

impl GraphSAGE<Untrained> {
    fn create_adjacency_matrix(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_nodes, _) = x.dim();
        let mut adjacency = Array2::zeros((n_nodes, n_nodes));
        let k = 10; // Number of neighbors

        for i in 0..n_nodes {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_nodes {
                if i != j {
                    let dist = x
                        .row(i)
                        .iter()
                        .zip(x.row(j).iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Connect to k nearest neighbors
            for &(j, _) in distances.iter().take(k) {
                adjacency[(i, j)] = 1.0;
            }
        }

        Ok(adjacency)
    }

    fn sage_layer_forward(
        &self,
        node_features: &Array2<f64>,
        adjacency: &Array2<f64>,
        weight: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_nodes = node_features.nrows();
        let feature_dim = node_features.ncols();

        // Sample and aggregate neighbors for each node
        let mut aggregated_features = Array2::zeros((n_nodes, feature_dim));

        for i in 0..n_nodes {
            // Find neighbors
            let neighbors: Vec<usize> = adjacency
                .row(i)
                .iter()
                .enumerate()
                .filter(|(_, &weight)| weight > 0.0)
                .map(|(idx, _)| idx)
                .collect();

            if !neighbors.is_empty() {
                // Sample neighbors if too many
                let sampled_neighbors = if neighbors.len() > self.sample_size {
                    let mut rng = thread_rng();
                    let mut sampled = neighbors;
                    sampled.shuffle(&mut rng);
                    sampled.into_iter().take(self.sample_size).collect()
                } else {
                    neighbors
                };

                // Aggregate neighbor features (mean aggregation)
                let mut neighbor_sum = Array1::zeros(feature_dim);
                for &neighbor in &sampled_neighbors {
                    neighbor_sum += &node_features.row(neighbor);
                }
                neighbor_sum /= sampled_neighbors.len() as f64;
                aggregated_features.row_mut(i).assign(&neighbor_sum);
            } else {
                // No neighbors, use own features
                aggregated_features.row_mut(i).assign(&node_features.row(i));
            }
        }

        // Concatenate self features with aggregated neighbor features
        let mut concatenated = Array2::zeros((n_nodes, feature_dim * 2));
        for i in 0..n_nodes {
            concatenated
                .slice_mut(s![i, ..feature_dim])
                .assign(&node_features.row(i));
            concatenated
                .slice_mut(s![i, feature_dim..])
                .assign(&aggregated_features.row(i));
        }

        // Apply linear transformation and activation
        let mut output = concatenated.dot(weight);
        output.mapv_inplace(|x| x.max(0.0)); // ReLU activation

        // L2 normalization
        for mut row in output.rows_mut() {
            let norm = row.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                row /= norm;
            }
        }

        Ok(output)
    }
}

/// Graph Attention Network (GAT)
///
/// Implements Graph Attention Networks with multi-head attention mechanism.
#[derive(Debug, Clone)]
pub struct GraphAttentionNetwork<S = Untrained> {
    state: S,
    n_components: usize,
    n_heads: usize,
    dropout_rate: f64,
    alpha: f64, // LeakyReLU negative slope
    learning_rate: f64,
    epochs: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct GATTrained {
    attention_weights: Vec<Array2<f64>>,
    weight_matrices: Vec<Array2<f64>>,
    embedding: Array2<f64>,
}

impl Default for GraphAttentionNetwork<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAttentionNetwork<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 64,
            n_heads: 8,
            dropout_rate: 0.6,
            alpha: 0.2,
            learning_rate: 0.005,
            epochs: 1000,
            random_state: None,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

impl Estimator for GraphAttentionNetwork<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for GraphAttentionNetwork<GATTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for GraphAttentionNetwork<Untrained> {
    type Fitted = GraphAttentionNetwork<GATTrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_nodes, n_features) = x.dim();

        // Create adjacency matrix
        let adjacency = self.create_adjacency_matrix(x)?;

        // Initialize weight matrices for multi-head attention
        let mut weight_matrices = Vec::new();
        let mut attention_weights = Vec::new();
        let mut rng = thread_rng();

        for _ in 0..self.n_heads {
            // Weight matrix for feature transformation
            let limit = (6.0 / (n_features + self.n_components) as f64).sqrt();
            let mut weight = Array2::zeros((n_features, self.n_components));
            for elem in weight.iter_mut() {
                *elem = rng.gen_range(-limit..limit);
            }
            weight_matrices.push(weight);

            // Attention mechanism parameters
            let mut attention = Array2::zeros((self.n_components * 2, 1));
            for elem in attention.iter_mut() {
                *elem = rng.gen_range(-0.1..0.1);
            }
            attention_weights.push(attention);
        }

        // Compute multi-head attention and aggregate
        let mut head_outputs = Vec::new();

        for head in 0..self.n_heads {
            let output = self.attention_head_forward(
                x,
                &adjacency,
                &weight_matrices[head],
                &attention_weights[head],
            )?;
            head_outputs.push(output);
        }

        // Concatenate or average the heads
        let embedding = if self.n_heads == 1 {
            head_outputs[0].clone()
        } else {
            // Average the heads
            let mut averaged = Array2::zeros((n_nodes, self.n_components));
            for output in &head_outputs {
                averaged += output;
            }
            averaged /= self.n_heads as f64;
            averaged
        };

        Ok(GraphAttentionNetwork {
            state: GATTrained {
                attention_weights,
                weight_matrices,
                embedding,
            },
            n_components: self.n_components,
            n_heads: self.n_heads,
            dropout_rate: self.dropout_rate,
            alpha: self.alpha,
            learning_rate: self.learning_rate,
            epochs: self.epochs,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for GraphAttentionNetwork<GATTrained> {
    fn transform(&self, _x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        Ok(self.state.embedding.clone())
    }
}

impl GraphAttentionNetwork<Untrained> {
    fn create_adjacency_matrix(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_nodes, _) = x.dim();
        let mut adjacency = Array2::zeros((n_nodes, n_nodes));
        let k = 10; // Number of neighbors

        for i in 0..n_nodes {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_nodes {
                if i != j {
                    let dist = x
                        .row(i)
                        .iter()
                        .zip(x.row(j).iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Connect to k nearest neighbors
            for &(j, _) in distances.iter().take(k) {
                adjacency[(i, j)] = 1.0;
            }
        }

        Ok(adjacency)
    }

    fn attention_head_forward(
        &self,
        features: &ArrayView2<f64>,
        adjacency: &Array2<f64>,
        weight: &Array2<f64>,
        attention_weight: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_nodes = features.nrows();

        // Linear transformation: h_i = W * x_i
        let transformed = features.dot(weight);

        // Compute attention coefficients
        let mut attention_scores = Array2::zeros((n_nodes, n_nodes));

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if adjacency[(i, j)] > 0.0 || i == j {
                    // Concatenate transformed features
                    let mut concat = Array1::zeros(self.n_components * 2);
                    concat
                        .slice_mut(s![..self.n_components])
                        .assign(&transformed.row(i));
                    concat
                        .slice_mut(s![self.n_components..])
                        .assign(&transformed.row(j));

                    // Compute attention score: e_ij = a^T [W*h_i || W*h_j]
                    let score = concat.dot(&attention_weight.column(0));

                    // Apply LeakyReLU
                    attention_scores[(i, j)] = if score > 0.0 {
                        score
                    } else {
                        self.alpha * score
                    };
                } else {
                    attention_scores[(i, j)] = f64::NEG_INFINITY;
                }
            }
        }

        // Apply softmax to get attention weights
        for mut row in attention_scores.rows_mut() {
            let max_val = row
                .iter()
                .filter(|&&x| x.is_finite())
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if max_val.is_finite() {
                for elem in row.iter_mut() {
                    if elem.is_finite() {
                        *elem = (*elem - max_val).exp();
                    } else {
                        *elem = 0.0;
                    }
                }
                let sum: f64 = row.sum();
                if sum > 1e-10 {
                    row /= sum;
                }
            }
        }

        // Apply attention to get final features
        let output = attention_scores.dot(&transformed);

        Ok(output)
    }
}
