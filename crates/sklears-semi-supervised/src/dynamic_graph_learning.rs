//! Dynamic graph learning for streaming and evolving semi-supervised scenarios
//!
//! This module provides advanced dynamic graph learning algorithms that can handle
//! continuously evolving graph structures, streaming data updates, and online
//! semi-supervised learning scenarios.

use scirs2_core::ndarray_ext::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rand_prelude::*;
use sklears_core::error::SklearsError;
use std::collections::{HashMap, VecDeque};

/// Dynamic graph learning for streaming and continuously evolving scenarios
#[derive(Clone)]
pub struct DynamicGraphLearning {
    /// Learning rate for online updates
    pub learning_rate: f64,
    /// Forgetting factor for old connections
    pub forgetting_factor: f64,
    /// Number of neighbors for new node integration
    pub k_neighbors: usize,
    /// Buffer size for streaming updates
    pub buffer_size: usize,
    /// Threshold for edge creation/removal
    pub edge_threshold: f64,
    /// Maximum number of nodes to maintain
    pub max_nodes: Option<usize>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Current adjacency matrix
    adjacency_matrix: Option<Array2<f64>>,
    /// Node features buffer
    node_features: Option<Array2<f64>>,
    /// Update history buffer
    update_buffer: VecDeque<GraphUpdate>,
}

/// Represents a graph update operation
#[derive(Clone, Debug)]
pub struct GraphUpdate {
    /// Type of update: "add_node", "remove_node", "update_edge", "update_features"
    pub update_type: String,
    /// Node indices involved
    pub node_indices: Vec<usize>,
    /// New feature values (for feature updates)
    pub features: Option<Array1<f64>>,
    /// Edge weight (for edge updates)
    pub edge_weight: Option<f64>,
    /// Timestamp of update
    pub timestamp: f64,
}

impl DynamicGraphLearning {
    /// Create a new dynamic graph learning instance
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            forgetting_factor: 0.95,
            k_neighbors: 5,
            buffer_size: 1000,
            edge_threshold: 0.1,
            max_nodes: None,
            random_state: None,
            adjacency_matrix: None,
            node_features: None,
            update_buffer: VecDeque::new(),
        }
    }

    /// Set the learning rate for online updates
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the forgetting factor for old connections
    pub fn forgetting_factor(mut self, factor: f64) -> Self {
        self.forgetting_factor = factor;
        self
    }

    /// Set the number of neighbors for new node integration
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the buffer size for streaming updates
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the edge threshold for creation/removal
    pub fn edge_threshold(mut self, threshold: f64) -> Self {
        self.edge_threshold = threshold;
        self
    }

    /// Set the maximum number of nodes to maintain
    pub fn max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = Some(max_nodes);
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Initialize the dynamic graph with initial data
    pub fn initialize(&mut self, initial_features: ArrayView2<f64>) -> Result<(), SklearsError> {
        let n_samples = initial_features.nrows();
        let n_features = initial_features.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No initial data provided".to_string(),
            ));
        }

        // Initialize node features
        self.node_features = Some(initial_features.to_owned());

        // Initialize adjacency matrix with k-NN graph
        let mut adjacency = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist =
                        self.compute_distance(initial_features.row(i), initial_features.row(j));
                    distances.push((j, dist));
                }
            }

            // Sort by distance and connect to k nearest neighbors
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(neighbor, dist) in distances.iter().take(self.k_neighbors) {
                let weight = (-dist).exp(); // Gaussian similarity
                adjacency[[i, neighbor]] = weight;
                adjacency[[neighbor, i]] = weight; // Symmetric
            }
        }

        self.adjacency_matrix = Some(adjacency);
        Ok(())
    }

    /// Add new nodes to the dynamic graph
    pub fn add_nodes(&mut self, new_features: ArrayView2<f64>) -> Result<(), SklearsError> {
        if self.node_features.is_none() || self.adjacency_matrix.is_none() {
            return Err(SklearsError::InvalidInput(
                "Graph not initialized".to_string(),
            ));
        }

        let new_n_nodes = new_features.nrows();

        // Check max nodes constraint and prune if necessary
        if let Some(max_nodes) = self.max_nodes {
            let current_n_nodes = self.node_features.as_ref().unwrap().nrows();
            let total_nodes = current_n_nodes + new_n_nodes;
            if total_nodes > max_nodes {
                self.prune_old_nodes(max_nodes - new_n_nodes)?;
            }
        }

        // Get references after potential pruning
        let current_features = self.node_features.as_ref().unwrap();
        let current_adjacency = self.adjacency_matrix.as_ref().unwrap();

        let old_n_nodes = current_features.nrows();
        let total_nodes = old_n_nodes + new_n_nodes;

        // Extend feature matrix
        let mut extended_features = Array2::zeros((total_nodes, current_features.ncols()));
        extended_features
            .slice_mut(s![..old_n_nodes, ..])
            .assign(current_features);
        extended_features
            .slice_mut(s![old_n_nodes.., ..])
            .assign(&new_features);

        // Extend adjacency matrix
        let mut extended_adjacency = Array2::zeros((total_nodes, total_nodes));
        extended_adjacency
            .slice_mut(s![..old_n_nodes, ..old_n_nodes])
            .assign(current_adjacency);

        // Connect new nodes to existing nodes
        for i in old_n_nodes..total_nodes {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..old_n_nodes {
                let dist =
                    self.compute_distance(extended_features.row(i), extended_features.row(j));
                distances.push((j, dist));
            }

            // Connect to k nearest existing neighbors
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(neighbor, dist) in distances.iter().take(self.k_neighbors) {
                let weight = (-dist).exp();
                extended_adjacency[[i, neighbor]] = weight;
                extended_adjacency[[neighbor, i]] = weight;
            }

            // Connect new nodes to each other
            for j in (old_n_nodes..total_nodes).filter(|&j| j != i) {
                let dist =
                    self.compute_distance(extended_features.row(i), extended_features.row(j));
                let weight = (-dist).exp();
                if weight > self.edge_threshold {
                    extended_adjacency[[i, j]] = weight;
                    extended_adjacency[[j, i]] = weight;
                }
            }
        }

        self.node_features = Some(extended_features);
        self.adjacency_matrix = Some(extended_adjacency);

        // Record updates
        for i in old_n_nodes..total_nodes {
            self.record_update(GraphUpdate {
                update_type: "add_node".to_string(),
                node_indices: vec![i],
                features: Some(new_features.row(i - old_n_nodes).to_owned()),
                edge_weight: None,
                timestamp: self.get_current_time(),
            });
        }

        Ok(())
    }

    /// Update node features dynamically
    pub fn update_node_features(
        &mut self,
        node_idx: usize,
        new_features: ArrayView1<f64>,
    ) -> Result<(), SklearsError> {
        if self.node_features.is_none() {
            return Err(SklearsError::InvalidInput(
                "Graph not initialized".to_string(),
            ));
        }

        let features = self.node_features.as_mut().unwrap();

        if node_idx >= features.nrows() {
            return Err(SklearsError::InvalidInput(
                "Node index out of bounds".to_string(),
            ));
        }

        // Apply online learning update
        let mut current_features = features.row_mut(node_idx);
        for (i, &new_val) in new_features.iter().enumerate() {
            current_features[i] =
                (1.0 - self.learning_rate) * current_features[i] + self.learning_rate * new_val;
        }

        // Update edges based on new features
        self.update_edges_for_node(node_idx)?;

        // Record update
        self.record_update(GraphUpdate {
            update_type: "update_features".to_string(),
            node_indices: vec![node_idx],
            features: Some(new_features.to_owned()),
            edge_weight: None,
            timestamp: self.get_current_time(),
        });

        Ok(())
    }

    /// Update edges for a specific node after feature change
    fn update_edges_for_node(&mut self, node_idx: usize) -> Result<(), SklearsError> {
        if self.node_features.is_none() || self.adjacency_matrix.is_none() {
            return Ok(());
        }

        // Create a copy of features to avoid borrowing conflicts
        let features = self.node_features.as_ref().unwrap().clone();
        let n_nodes = features.nrows();
        let forgetting_factor = self.forgetting_factor;
        let edge_threshold = self.edge_threshold;

        // Get mutable reference to adjacency matrix
        let adjacency = self.adjacency_matrix.as_mut().unwrap();

        // Recompute edges for this node
        for other_idx in 0..n_nodes {
            if node_idx != other_idx {
                let dist =
                    Self::compute_distance_static(features.row(node_idx), features.row(other_idx));
                let new_weight = (-dist).exp();

                // Apply forgetting factor to existing edge and add new weight
                let current_weight = adjacency[[node_idx, other_idx]];
                let updated_weight =
                    forgetting_factor * current_weight + (1.0 - forgetting_factor) * new_weight;

                // Apply threshold for edge maintenance
                let final_weight = if updated_weight > edge_threshold {
                    updated_weight
                } else {
                    0.0
                };

                adjacency[[node_idx, other_idx]] = final_weight;
                adjacency[[other_idx, node_idx]] = final_weight; // Symmetric
            }
        }

        Ok(())
    }

    /// Prune old nodes to maintain memory constraints
    fn prune_old_nodes(&mut self, target_nodes: usize) -> Result<(), SklearsError> {
        if self.node_features.is_none() || self.adjacency_matrix.is_none() {
            return Ok(());
        }

        let current_nodes = self.node_features.as_ref().unwrap().nrows();
        if current_nodes <= target_nodes {
            return Ok(());
        }

        let nodes_to_remove = current_nodes - target_nodes;

        // Simple strategy: remove oldest nodes (first nodes_to_remove nodes)
        // In practice, you might want more sophisticated strategies based on
        // node importance, connectivity, or recency of updates

        let features = self.node_features.as_ref().unwrap();
        let adjacency = self.adjacency_matrix.as_ref().unwrap();

        // Create new matrices without the pruned nodes
        let new_features = features.slice(s![nodes_to_remove.., ..]).to_owned();
        let new_adjacency = adjacency
            .slice(s![nodes_to_remove.., nodes_to_remove..])
            .to_owned();

        self.node_features = Some(new_features);
        self.adjacency_matrix = Some(new_adjacency);

        Ok(())
    }

    /// Get the current adjacency matrix
    pub fn get_adjacency_matrix(&self) -> Option<&Array2<f64>> {
        self.adjacency_matrix.as_ref()
    }

    /// Get the current node features
    pub fn get_node_features(&self) -> Option<&Array2<f64>> {
        self.node_features.as_ref()
    }

    /// Get recent updates from the buffer
    pub fn get_recent_updates(&self, n_updates: usize) -> Vec<&GraphUpdate> {
        self.update_buffer.iter().rev().take(n_updates).collect()
    }

    /// Compute distance between two feature vectors
    fn compute_distance(&self, feat1: ArrayView1<f64>, feat2: ArrayView1<f64>) -> f64 {
        Self::compute_distance_static(feat1, feat2)
    }

    /// Static version of compute_distance to avoid borrowing conflicts
    fn compute_distance_static(feat1: ArrayView1<f64>, feat2: ArrayView1<f64>) -> f64 {
        feat1
            .iter()
            .zip(feat2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Record a graph update in the buffer
    fn record_update(&mut self, update: GraphUpdate) {
        self.update_buffer.push_back(update);

        // Maintain buffer size
        while self.update_buffer.len() > self.buffer_size {
            self.update_buffer.pop_front();
        }
    }

    /// Get current timestamp (simplified)
    fn get_current_time(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    /// Apply decay to all edges to simulate forgetting
    pub fn apply_temporal_decay(&mut self) -> Result<(), SklearsError> {
        if let Some(adjacency) = self.adjacency_matrix.as_mut() {
            *adjacency *= self.forgetting_factor;

            // Remove edges below threshold
            adjacency.mapv_inplace(|x| if x < self.edge_threshold { 0.0 } else { x });
        }
        Ok(())
    }

    /// Get graph statistics
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if let Some(adjacency) = &self.adjacency_matrix {
            let n_nodes = adjacency.nrows() as f64;
            let total_edges = adjacency.iter().filter(|&&x| x > 0.0).count() as f64 / 2.0; // Undirected
            let density = if n_nodes > 1.0 {
                total_edges / (n_nodes * (n_nodes - 1.0) / 2.0)
            } else {
                0.0
            };

            stats.insert("n_nodes".to_string(), n_nodes);
            stats.insert("n_edges".to_string(), total_edges);
            stats.insert("density".to_string(), density);
            stats.insert("avg_degree".to_string(), total_edges * 2.0 / n_nodes);
        }

        stats.insert("buffer_size".to_string(), self.update_buffer.len() as f64);
        stats
    }
}

impl Default for DynamicGraphLearning {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    fn test_dynamic_graph_initialization() {
        let mut dgl = DynamicGraphLearning::new().k_neighbors(2);

        let initial_data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let result = dgl.initialize(initial_data.view());
        assert!(result.is_ok());

        let adjacency = dgl.get_adjacency_matrix().unwrap();
        assert_eq!(adjacency.dim(), (3, 3));

        // Check that diagonal is zero
        for i in 0..3 {
            assert_eq!(adjacency[[i, i]], 0.0);
        }
    }

    #[test]
    fn test_add_nodes() {
        let mut dgl = DynamicGraphLearning::new().k_neighbors(2);

        let initial_data = array![[1.0, 2.0], [2.0, 3.0]];

        dgl.initialize(initial_data.view()).unwrap();

        let new_data = array![[3.0, 4.0], [4.0, 5.0]];

        let result = dgl.add_nodes(new_data.view());
        assert!(result.is_ok());

        let adjacency = dgl.get_adjacency_matrix().unwrap();
        assert_eq!(adjacency.dim(), (4, 4));

        let features = dgl.get_node_features().unwrap();
        assert_eq!(features.dim(), (4, 2));
    }

    #[test]
    fn test_update_node_features() {
        let mut dgl = DynamicGraphLearning::new()
            .k_neighbors(2)
            .learning_rate(0.5);

        let initial_data = array![[1.0, 2.0], [2.0, 3.0]];

        dgl.initialize(initial_data.view()).unwrap();

        let new_features = array![5.0, 6.0];
        let result = dgl.update_node_features(0, new_features.view());
        assert!(result.is_ok());

        let features = dgl.get_node_features().unwrap();
        // Features should be updated with learning rate
        assert!(features[[0, 0]] > 1.0);
        assert!(features[[0, 1]] > 2.0);
    }

    #[test]
    fn test_temporal_decay() {
        let mut dgl = DynamicGraphLearning::new()
            .k_neighbors(2)
            .forgetting_factor(0.5)
            .edge_threshold(0.1);

        let initial_data = array![[1.0, 2.0], [2.0, 3.0]];

        dgl.initialize(initial_data.view()).unwrap();

        let original_adjacency = dgl.get_adjacency_matrix().unwrap().clone();

        dgl.apply_temporal_decay().unwrap();

        let decayed_adjacency = dgl.get_adjacency_matrix().unwrap();

        // Check that edges have been decayed
        for i in 0..2 {
            for j in 0..2 {
                if i != j && original_adjacency[[i, j]] > 0.0 {
                    assert!(decayed_adjacency[[i, j]] < original_adjacency[[i, j]]);
                }
            }
        }
    }

    #[test]
    fn test_max_nodes_constraint() {
        let mut dgl = DynamicGraphLearning::new().k_neighbors(2).max_nodes(3);

        let initial_data = array![[1.0, 2.0], [2.0, 3.0]];

        dgl.initialize(initial_data.view()).unwrap();

        let new_data = array![[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];

        let result = dgl.add_nodes(new_data.view());
        assert!(result.is_ok());

        let adjacency = dgl.get_adjacency_matrix().unwrap();
        assert_eq!(adjacency.nrows(), 3); // Should be pruned to max_nodes
    }

    #[test]
    fn test_graph_statistics() {
        let mut dgl = DynamicGraphLearning::new().k_neighbors(2);

        let initial_data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        dgl.initialize(initial_data.view()).unwrap();

        let stats = dgl.get_statistics();

        assert!(stats.contains_key("n_nodes"));
        assert!(stats.contains_key("n_edges"));
        assert!(stats.contains_key("density"));
        assert!(stats.contains_key("avg_degree"));

        assert_eq!(stats["n_nodes"], 3.0);
        assert!(stats["n_edges"] > 0.0);
    }

    #[test]
    fn test_update_buffer() {
        let mut dgl = DynamicGraphLearning::new().buffer_size(2);

        let initial_data = array![[1.0, 2.0], [2.0, 3.0]];

        dgl.initialize(initial_data.view()).unwrap();

        let new_features = array![5.0, 6.0];
        dgl.update_node_features(0, new_features.view()).unwrap();
        dgl.update_node_features(1, new_features.view()).unwrap();
        dgl.update_node_features(0, new_features.view()).unwrap();

        let recent_updates = dgl.get_recent_updates(5);
        assert!(recent_updates.len() <= 2); // Buffer size constraint
    }

    #[test]
    fn test_error_cases() {
        let mut dgl = DynamicGraphLearning::new();

        // Test operations before initialization
        let new_data = array![[1.0, 2.0]];
        assert!(dgl.add_nodes(new_data.view()).is_err());

        let new_features = array![5.0, 6.0];
        assert!(dgl.update_node_features(0, new_features.view()).is_err());

        // Test initialization with empty data
        let empty_data = Array2::<f64>::zeros((0, 2));
        assert!(dgl.initialize(empty_data.view()).is_err());

        // Test feature update with invalid index
        let initial_data = array![[1.0, 2.0]];
        dgl.initialize(initial_data.view()).unwrap();
        assert!(dgl.update_node_features(10, new_features.view()).is_err());
    }
}
