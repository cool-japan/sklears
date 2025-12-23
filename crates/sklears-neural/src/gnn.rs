//! Graph Neural Networks for learning on graph-structured data.
//!
//! This module implements various graph neural network architectures including:
//! - Graph Convolutional Networks (GCN)
//! - Graph Attention Networks (GAT)
//! - Message Passing Neural Networks (MPNN)
//! - GraphSAGE (SAmple and aggreGatE)
//! - Graph Isomorphism Networks (GIN)
//! - Graph pooling operations
//!
//! Graph Neural Networks learn representations of nodes, edges, and entire graphs
//! by propagating information along the graph structure.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Axis, ScalarOperand};
use scirs2_core::random::{thread_rng, CoreRandom, Normal, Rng};
use sklears_core::{error::SklearsError, types::FloatBounds};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Graph representation with adjacency information
#[derive(Debug, Clone)]
pub struct Graph<T: FloatBounds> {
    /// Node features (num_nodes × feature_dim)
    pub node_features: Array2<T>,
    /// Edge list: list of (source, target) pairs
    pub edge_list: Vec<(usize, usize)>,
    /// Edge features (optional)
    pub edge_features: Option<Array2<T>>,
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
}

impl<T: FloatBounds> Graph<T> {
    /// Create a new graph
    pub fn new(
        node_features: Array2<T>,
        edge_list: Vec<(usize, usize)>,
        edge_features: Option<Array2<T>>,
    ) -> NeuralResult<Self> {
        let num_nodes = node_features.nrows();
        let num_edges = edge_list.len();

        // Validate edge list
        for &(src, dst) in &edge_list {
            if src >= num_nodes || dst >= num_nodes {
                return Err(SklearsError::InvalidParameter {
                    name: "edge_list".to_string(),
                    reason: format!(
                        "Edge ({}, {}) references non-existent node (max index: {})",
                        src,
                        dst,
                        num_nodes - 1
                    ),
                });
            }
        }

        // Validate edge features
        if let Some(ref features) = edge_features {
            if features.nrows() != num_edges {
                return Err(SklearsError::InvalidParameter {
                    name: "edge_features".to_string(),
                    reason: format!(
                        "Number of edge features ({}) doesn't match number of edges ({})",
                        features.nrows(),
                        num_edges
                    ),
                });
            }
        }

        Ok(Self {
            node_features,
            edge_list,
            edge_features,
            num_nodes,
            num_edges,
        })
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edge_list
            .iter()
            .filter(|&&(src, _)| src == node)
            .map(|&(_, dst)| dst)
            .collect()
    }

    /// Compute degree for each node
    pub fn node_degrees(&self) -> Vec<usize> {
        let mut degrees = vec![0; self.num_nodes];
        for &(src, _) in &self.edge_list {
            degrees[src] += 1;
        }
        degrees
    }

    /// Build adjacency matrix
    pub fn adjacency_matrix(&self) -> Array2<T> {
        let mut adj = Array2::zeros((self.num_nodes, self.num_nodes));
        for &(src, dst) in &self.edge_list {
            adj[[src, dst]] = T::one();
        }
        adj
    }

    /// Add self-loops to the graph
    pub fn add_self_loops(&mut self) {
        let mut new_edges = HashSet::new();
        for edge in &self.edge_list {
            new_edges.insert(*edge);
        }

        for i in 0..self.num_nodes {
            new_edges.insert((i, i));
        }

        self.edge_list = new_edges.into_iter().collect();
        self.num_edges = self.edge_list.len();
    }
}

/// Aggregation function for message passing
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AggregationType {
    /// Sum aggregation
    Sum,
    /// Mean aggregation
    Mean,
    /// Max aggregation
    Max,
    /// Attention-weighted aggregation
    Attention,
}

/// Graph Convolutional Network (GCN) layer
///
/// Implements the GCN propagation rule:
/// H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
#[derive(Debug)]
pub struct GCNLayer<T: FloatBounds> {
    /// Weight matrix
    weight: Array2<T>,
    /// Bias
    bias: Option<Array1<T>>,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
    /// Whether to use bias
    use_bias: bool,
    /// Cached values for backward pass
    cached_input: Option<Array2<T>>,
    cached_adj_norm: Option<Array2<T>>,
}

impl<T: FloatBounds + ScalarOperand> GCNLayer<T> {
    /// Create a new GCN layer
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let mut rng = thread_rng();
        let std = (2.0 / in_features as f64).sqrt();

        let weight = Array2::from_shape_fn((in_features, out_features), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std).unwrap()
        });

        let bias = if use_bias {
            Some(Array1::zeros(out_features))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
            out_features,
            use_bias,
            cached_input: None,
            cached_adj_norm: None,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array2<T>, graph: &Graph<T>) -> NeuralResult<Array2<T>> {
        // Build normalized adjacency matrix with self-loops
        let mut adj = graph.adjacency_matrix();

        // Add self-loops
        for i in 0..graph.num_nodes {
            adj[[i, i]] = adj[[i, i]] + T::one();
        }

        // Compute degree matrix D^(-1/2)
        let degrees = adj.sum_axis(Axis(1));
        let mut d_inv_sqrt = Array2::zeros((graph.num_nodes, graph.num_nodes));
        for i in 0..graph.num_nodes {
            if degrees[i] > T::zero() {
                d_inv_sqrt[[i, i]] = degrees[i].powf(T::from(-0.5).unwrap());
            }
        }

        // Normalized adjacency: D^(-1/2) A D^(-1/2)
        let adj_norm = d_inv_sqrt.dot(&adj).dot(&d_inv_sqrt);

        // Message passing: A_norm @ X @ W
        let mut h = adj_norm.dot(x).dot(&self.weight);

        // Add bias
        if let Some(ref b) = self.bias {
            for i in 0..h.nrows() {
                h.row_mut(i).scaled_add(T::one(), &b.view());
            }
        }

        // Cache for backward pass
        self.cached_input = Some(x.clone());
        self.cached_adj_norm = Some(adj_norm);

        Ok(h)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weight.len() + self.bias.as_ref().map(|b| b.len()).unwrap_or(0)
    }
}

/// Graph Attention Network (GAT) layer
///
/// Implements multi-head attention mechanism for graphs:
/// α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
/// h_i' = σ(Σ_j α_ij W h_j)
#[derive(Debug)]
pub struct GATLayer<T: FloatBounds> {
    /// Weight matrix for features
    weight: Array2<T>,
    /// Attention weights (left)
    attention_left: Array1<T>,
    /// Attention weights (right)
    attention_right: Array1<T>,
    /// Number of attention heads
    num_heads: usize,
    /// Input dimension
    in_features: usize,
    /// Output dimension per head
    out_features_per_head: usize,
    /// LeakyReLU negative slope
    alpha: T,
    /// Dropout rate (placeholder for future implementation)
    dropout: f64,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum> GATLayer<T> {
    /// Create a new GAT layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        alpha: T,
        dropout: f64,
    ) -> Self {
        let mut rng = thread_rng();
        let out_features_per_head = out_features / num_heads;
        let total_out = out_features_per_head * num_heads;

        let std = (2.0 / in_features as f64).sqrt();
        let weight = Array2::from_shape_fn((in_features, total_out), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std).unwrap()
        });

        let attention_left = Array1::from_shape_fn(total_out, |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std).unwrap()
        });

        let attention_right = Array1::from_shape_fn(total_out, |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std).unwrap()
        });

        Self {
            weight,
            attention_left,
            attention_right,
            num_heads,
            in_features,
            out_features_per_head,
            alpha,
            dropout,
        }
    }

    /// Forward pass with attention
    pub fn forward(&mut self, x: &Array2<T>, graph: &Graph<T>) -> NeuralResult<Array2<T>> {
        let num_nodes = x.nrows();

        // Linear transformation: H' = XW
        let h_prime = x.dot(&self.weight);

        // Compute attention coefficients
        let mut attention_logits = Array2::zeros((num_nodes, num_nodes));

        for &(src, dst) in &graph.edge_list {
            // Attention mechanism: a^T [Wh_i || Wh_j]
            let h_src = h_prime.row(src);
            let h_dst = h_prime.row(dst);

            let attention_src = h_src.dot(&self.attention_left);
            let attention_dst = h_dst.dot(&self.attention_right);

            let attention_score = attention_src + attention_dst;

            // LeakyReLU
            let leak_value = if attention_score > T::zero() {
                attention_score
            } else {
                attention_score * self.alpha
            };

            attention_logits[[src, dst]] = leak_value;
        }

        // Apply softmax per node
        let mut attention_weights = Array2::zeros((num_nodes, num_nodes));
        for i in 0..num_nodes {
            let neighbors: Vec<usize> = graph
                .edge_list
                .iter()
                .filter(|&&(src, _)| src == i)
                .map(|&(_, dst)| dst)
                .collect();

            if neighbors.is_empty() {
                continue;
            }

            // Compute softmax over neighbors
            let max_logit = neighbors
                .iter()
                .map(|&j| attention_logits[[i, j]])
                .max_by(|a, b| {
                    a.to_f64()
                        .unwrap()
                        .partial_cmp(&b.to_f64().unwrap())
                        .unwrap()
                })
                .unwrap();

            let exp_sum: T = neighbors
                .iter()
                .map(|&j| (attention_logits[[i, j]] - max_logit).exp())
                .sum();

            for &j in &neighbors {
                attention_weights[[i, j]] = (attention_logits[[i, j]] - max_logit).exp() / exp_sum;
            }
        }

        // Apply attention weights to aggregate neighbors
        let mut output = Array2::zeros((num_nodes, self.out_features_per_head * self.num_heads));
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if attention_weights[[i, j]] > T::zero() {
                    for k in 0..output.ncols() {
                        output[[i, k]] =
                            output[[i, k]] + attention_weights[[i, j]] * h_prime[[j, k]];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weight.len() + self.attention_left.len() + self.attention_right.len()
    }
}

/// GraphSAGE layer for inductive learning
///
/// Aggregates neighbor features using various aggregation functions
#[derive(Debug)]
pub struct GraphSAGELayer<T: FloatBounds> {
    /// Weight matrix for self features
    weight_self: Array2<T>,
    /// Weight matrix for neighbor features
    weight_neighbor: Array2<T>,
    /// Aggregation type
    aggregation: AggregationType,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
    /// Whether to normalize output
    normalize: bool,
}

impl<T: FloatBounds + ScalarOperand> GraphSAGELayer<T> {
    /// Create a new GraphSAGE layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        aggregation: AggregationType,
        normalize: bool,
    ) -> Self {
        let mut rng = thread_rng();
        let std = (2.0 / in_features as f64).sqrt();

        let weight_self = Array2::from_shape_fn((in_features, out_features), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std).unwrap()
        });

        let weight_neighbor = Array2::from_shape_fn((in_features, out_features), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std).unwrap()
        });

        Self {
            weight_self,
            weight_neighbor,
            aggregation,
            in_features,
            out_features,
            normalize,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array2<T>, graph: &Graph<T>) -> NeuralResult<Array2<T>> {
        let num_nodes = x.nrows();

        // Aggregate neighbor features
        let mut aggregated = Array2::zeros((num_nodes, self.in_features));

        for i in 0..num_nodes {
            let neighbors = graph.neighbors(i);

            if neighbors.is_empty() {
                // If no neighbors, use self features
                for j in 0..self.in_features {
                    aggregated[[i, j]] = x[[i, j]];
                }
                continue;
            }

            match self.aggregation {
                AggregationType::Mean => {
                    // Mean aggregation
                    for &neighbor in &neighbors {
                        for j in 0..self.in_features {
                            aggregated[[i, j]] = aggregated[[i, j]] + x[[neighbor, j]];
                        }
                    }
                    let n_neighbors = T::from(neighbors.len() as f64).unwrap();
                    for j in 0..self.in_features {
                        aggregated[[i, j]] = aggregated[[i, j]] / n_neighbors;
                    }
                }
                AggregationType::Sum => {
                    // Sum aggregation
                    for &neighbor in &neighbors {
                        for j in 0..self.in_features {
                            aggregated[[i, j]] = aggregated[[i, j]] + x[[neighbor, j]];
                        }
                    }
                }
                AggregationType::Max => {
                    // Max aggregation
                    for j in 0..self.in_features {
                        let max_val = neighbors
                            .iter()
                            .map(|&n| x[[n, j]])
                            .max_by(|a, b| {
                                a.to_f64()
                                    .unwrap()
                                    .partial_cmp(&b.to_f64().unwrap())
                                    .unwrap()
                            })
                            .unwrap();
                        aggregated[[i, j]] = max_val;
                    }
                }
                AggregationType::Attention => {
                    // Simplified attention aggregation
                    for &neighbor in &neighbors {
                        for j in 0..self.in_features {
                            aggregated[[i, j]] = aggregated[[i, j]] + x[[neighbor, j]];
                        }
                    }
                    let n_neighbors = T::from(neighbors.len() as f64).unwrap();
                    for j in 0..self.in_features {
                        aggregated[[i, j]] = aggregated[[i, j]] / n_neighbors;
                    }
                }
            }
        }

        // Combine self and neighbor features
        let h_self = x.dot(&self.weight_self);
        let h_neighbor = aggregated.dot(&self.weight_neighbor);
        let mut output = h_self + h_neighbor;

        // Apply ReLU
        output.mapv_inplace(|x| if x > T::zero() { x } else { T::zero() });

        // Normalize if configured
        if self.normalize {
            for i in 0..num_nodes {
                let norm = output
                    .row(i)
                    .mapv(|x| x * x)
                    .sum()
                    .sqrt()
                    .max(T::from(1e-12).unwrap());
                for j in 0..self.out_features {
                    output[[i, j]] = output[[i, j]] / norm;
                }
            }
        }

        Ok(output)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weight_self.len() + self.weight_neighbor.len()
    }
}

/// Graph Isomorphism Network (GIN) layer
///
/// More expressive than GCN by using injective aggregation
#[derive(Debug)]
pub struct GINLayer<T: FloatBounds> {
    /// Epsilon (learnable parameter)
    epsilon: T,
    /// MLP weights
    mlp_weights: Vec<Array2<T>>,
    /// MLP biases
    mlp_biases: Vec<Array1<T>>,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
}

impl<T: FloatBounds + ScalarOperand> GINLayer<T> {
    /// Create a new GIN layer
    pub fn new(in_features: usize, out_features: usize, hidden_dim: usize) -> Self {
        let mut rng = thread_rng();

        // Initialize epsilon
        let epsilon = T::zero();

        // Two-layer MLP
        let mut mlp_weights = Vec::new();
        let mut mlp_biases = Vec::new();

        // First layer
        let std = (2.0 / in_features as f64).sqrt();
        let w1 = Array2::from_shape_fn((in_features, hidden_dim), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std).unwrap()
        });
        let b1 = Array1::zeros(hidden_dim);
        mlp_weights.push(w1);
        mlp_biases.push(b1);

        // Second layer
        let std = (2.0 / hidden_dim as f64).sqrt();
        let w2 = Array2::from_shape_fn((hidden_dim, out_features), |_| {
            T::from(rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap()) * std).unwrap()
        });
        let b2 = Array1::zeros(out_features);
        mlp_weights.push(w2);
        mlp_biases.push(b2);

        Self {
            epsilon,
            mlp_weights,
            mlp_biases,
            in_features,
            out_features,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array2<T>, graph: &Graph<T>) -> NeuralResult<Array2<T>> {
        let num_nodes = x.nrows();

        // Aggregate: (1 + ε) * h_i + Σ_j h_j
        let mut aggregated = Array2::zeros((num_nodes, self.in_features));

        for i in 0..num_nodes {
            // Add self features with (1 + epsilon)
            for j in 0..self.in_features {
                aggregated[[i, j]] = (T::one() + self.epsilon) * x[[i, j]];
            }

            // Add neighbor features
            let neighbors = graph.neighbors(i);
            for &neighbor in &neighbors {
                for j in 0..self.in_features {
                    aggregated[[i, j]] = aggregated[[i, j]] + x[[neighbor, j]];
                }
            }
        }

        // Apply MLP
        let mut h = aggregated;
        for (i, (w, b)) in self
            .mlp_weights
            .iter()
            .zip(self.mlp_biases.iter())
            .enumerate()
        {
            h = h.dot(w);
            for j in 0..h.nrows() {
                h.row_mut(j).scaled_add(T::one(), &b.view());
            }

            // ReLU for all but last layer
            if i < self.mlp_weights.len() - 1 {
                h.mapv_inplace(|x| if x > T::zero() { x } else { T::zero() });
            }
        }

        Ok(h)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.mlp_weights.iter().map(|w| w.len()).sum::<usize>()
            + self.mlp_biases.iter().map(|b| b.len()).sum::<usize>()
            + 1 // epsilon
    }
}

/// Graph-level pooling operations
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GraphPooling {
    /// Global mean pooling
    Mean,
    /// Global max pooling
    Max,
    /// Global sum pooling
    Sum,
    /// Global attention pooling
    Attention,
}

/// Apply graph pooling to get graph-level representation
pub fn pool_graph<T: FloatBounds + ScalarOperand>(
    node_features: &Array2<T>,
    pooling: GraphPooling,
) -> Array1<T> {
    match pooling {
        GraphPooling::Mean => node_features.mean_axis(Axis(0)).unwrap(),
        GraphPooling::Sum => node_features.sum_axis(Axis(0)),
        GraphPooling::Max => {
            let mut result = Array1::zeros(node_features.ncols());
            for j in 0..node_features.ncols() {
                let column = node_features.column(j);
                let max_val = column
                    .iter()
                    .max_by(|a, b| {
                        a.to_f64()
                            .unwrap()
                            .partial_cmp(&b.to_f64().unwrap())
                            .unwrap()
                    })
                    .unwrap();
                result[j] = *max_val;
            }
            result
        }
        GraphPooling::Attention => {
            // Simplified attention pooling (equal weights for now)
            node_features.mean_axis(Axis(0)).unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_graph() -> Graph<f64> {
        let node_features = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)];

        Graph::new(node_features, edge_list, None).unwrap()
    }

    #[test]
    fn test_graph_creation() {
        let graph = create_test_graph();
        assert_eq!(graph.num_nodes, 4);
        assert_eq!(graph.num_edges, 5);
    }

    #[test]
    fn test_graph_neighbors() {
        let graph = create_test_graph();
        let neighbors = graph.neighbors(1);
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
    }

    #[test]
    fn test_graph_degrees() {
        let graph = create_test_graph();
        let degrees = graph.node_degrees();
        assert_eq!(degrees[0], 1); // Node 0 has 1 outgoing edge
        assert_eq!(degrees[1], 2); // Node 1 has 2 outgoing edges
    }

    #[test]
    fn test_adjacency_matrix() {
        let graph = create_test_graph();
        let adj = graph.adjacency_matrix();

        assert_eq!(adj[[0, 1]], 1.0);
        assert_eq!(adj[[1, 2]], 1.0);
        assert_eq!(adj[[0, 2]], 0.0); // No edge
    }

    #[test]
    fn test_gcn_layer_creation() {
        let layer: GCNLayer<f64> = GCNLayer::new(10, 16, true);
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 16);
        assert!(layer.num_parameters() > 0);
    }

    #[test]
    fn test_gcn_layer_forward() {
        let mut layer: GCNLayer<f64> = GCNLayer::new(3, 8, true);
        let graph = create_test_graph();

        let output = layer.forward(&graph.node_features, &graph).unwrap();
        assert_eq!(output.nrows(), 4);
        assert_eq!(output.ncols(), 8);
    }

    #[test]
    fn test_gat_layer_creation() {
        let layer: GATLayer<f64> = GATLayer::new(10, 16, 2, 0.2, 0.6);
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.num_heads, 2);
        assert!(layer.num_parameters() > 0);
    }

    #[test]
    fn test_gat_layer_forward() {
        let mut layer: GATLayer<f64> = GATLayer::new(3, 8, 2, 0.2, 0.0);
        let graph = create_test_graph();

        let output = layer.forward(&graph.node_features, &graph).unwrap();
        assert_eq!(output.nrows(), 4);
        assert_eq!(output.ncols(), 8);
    }

    #[test]
    fn test_graphsage_layer_creation() {
        let layer: GraphSAGELayer<f64> = GraphSAGELayer::new(10, 16, AggregationType::Mean, true);
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 16);
        assert!(layer.num_parameters() > 0);
    }

    #[test]
    fn test_graphsage_layer_forward() {
        let mut layer: GraphSAGELayer<f64> =
            GraphSAGELayer::new(3, 8, AggregationType::Mean, false);
        let graph = create_test_graph();

        let output = layer.forward(&graph.node_features, &graph).unwrap();
        assert_eq!(output.nrows(), 4);
        assert_eq!(output.ncols(), 8);
    }

    #[test]
    fn test_gin_layer_creation() {
        let layer: GINLayer<f64> = GINLayer::new(10, 16, 32);
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 16);
        assert!(layer.num_parameters() > 0);
    }

    #[test]
    fn test_gin_layer_forward() {
        let mut layer: GINLayer<f64> = GINLayer::new(3, 8, 16);
        let graph = create_test_graph();

        let output = layer.forward(&graph.node_features, &graph).unwrap();
        assert_eq!(output.nrows(), 4);
        assert_eq!(output.ncols(), 8);
    }

    #[test]
    fn test_graph_pooling_mean() {
        let graph = create_test_graph();
        let pooled = pool_graph(&graph.node_features, GraphPooling::Mean);

        assert_eq!(pooled.len(), 3);
        assert_relative_eq!(pooled[0], 5.5, epsilon = 1e-6); // Mean of [1, 4, 7, 10]
    }

    #[test]
    fn test_graph_pooling_max() {
        let graph = create_test_graph();
        let pooled = pool_graph(&graph.node_features, GraphPooling::Max);

        assert_eq!(pooled.len(), 3);
        assert_eq!(pooled[0], 10.0); // Max of [1, 4, 7, 10]
    }

    #[test]
    fn test_graph_pooling_sum() {
        let graph = create_test_graph();
        let pooled = pool_graph(&graph.node_features, GraphPooling::Sum);

        assert_eq!(pooled.len(), 3);
        assert_eq!(pooled[0], 22.0); // Sum of [1, 4, 7, 10]
    }

    #[test]
    fn test_add_self_loops() {
        let mut graph = create_test_graph();
        let original_edges = graph.num_edges;

        graph.add_self_loops();

        // Should have added self-loops for each node
        assert!(graph.num_edges >= original_edges);
    }

    #[test]
    fn test_graphsage_max_aggregation() {
        let mut layer: GraphSAGELayer<f64> = GraphSAGELayer::new(3, 8, AggregationType::Max, false);
        let graph = create_test_graph();

        let output = layer.forward(&graph.node_features, &graph).unwrap();
        assert_eq!(output.dim(), (4, 8));
    }

    #[test]
    fn test_graphsage_sum_aggregation() {
        let mut layer: GraphSAGELayer<f64> = GraphSAGELayer::new(3, 8, AggregationType::Sum, false);
        let graph = create_test_graph();

        let output = layer.forward(&graph.node_features, &graph).unwrap();
        assert_eq!(output.dim(), (4, 8));
    }
}
