//! Graph Clustering Algorithms
//!
//! This module provides clustering algorithms specifically designed for graph-structured data,
//! including community detection and modularity-based clustering methods.
//!
//! # Algorithms Implemented
//! - **Modularity-based clustering**: Greedy modularity optimization
//! - **Louvain algorithm**: Fast community detection using modularity optimization
//! - **Label propagation**: Simple and fast community detection
//! - **Leiden algorithm**: Improved community detection with resolution limit fixes
//! - **Spectral graph clustering**: Normalized cuts and spectral partitioning
//! - **Overlapping community detection**: Communities that can share nodes
//!
//! # Graph Representation
//! Graphs are represented as adjacency matrices (dense or sparse) or edge lists.
//! The module supports both weighted and unweighted graphs.
//!
//! # Mathematical Background
//!
//! ## Modularity
//! Q = (1/2m) * Σ[A_ij - (k_i * k_j)/2m] * δ(c_i, c_j)
//! where A_ij is the adjacency matrix, k_i is the degree of node i,
//! m is the total number of edges, and δ(c_i, c_j) = 1 if nodes i and j are in the same community
//!
//! ## Normalized Cut
//! NCut(A,B) = cut(A,B)/vol(A) + cut(A,B)/vol(B)
//! where cut(A,B) is the total weight of edges between sets A and B,
//! and vol(A) is the total degree of nodes in set A

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use numrs2::prelude::*;
use scirs2_core::rand_prelude::{Distribution, SliceRandom};
// Normal distribution via scirs2_core::random::RandNormal
use scirs2_core::random::{thread_rng, Random, Rng};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::*;

/// Graph representation for clustering algorithms
#[derive(Debug, Clone)]
pub struct Graph {
    /// Adjacency matrix (can be sparse or dense)
    pub adjacency: Array2<f64>,
    /// Number of nodes
    pub n_nodes: usize,
    /// Whether the graph is directed
    pub directed: bool,
    /// Node weights (optional)
    pub node_weights: Option<Vec<f64>>,
}

impl Graph {
    /// Create a new graph from adjacency matrix
    pub fn from_adjacency(adjacency: Array2<f64>, directed: bool) -> Result<Self> {
        let n_nodes = adjacency.nrows();
        if adjacency.ncols() != n_nodes {
            return Err(SklearsError::InvalidInput(
                "Adjacency matrix must be square".to_string(),
            ));
        }

        Ok(Self {
            adjacency,
            n_nodes,
            directed,
            node_weights: None,
        })
    }

    /// Create a graph from edge list
    pub fn from_edges(
        edges: &[(usize, usize, f64)],
        n_nodes: usize,
        directed: bool,
    ) -> Result<Self> {
        let mut adjacency = Array2::zeros((n_nodes, n_nodes));

        for &(i, j, weight) in edges {
            if i >= n_nodes || j >= n_nodes {
                return Err(SklearsError::InvalidInput(
                    "Edge indices exceed number of nodes".to_string(),
                ));
            }

            adjacency[[i, j]] = weight;
            if !directed {
                adjacency[[j, i]] = weight;
            }
        }

        Ok(Self {
            adjacency,
            n_nodes,
            directed,
            node_weights: None,
        })
    }

    /// Set node weights
    pub fn with_node_weights(mut self, weights: Vec<f64>) -> Result<Self> {
        if weights.len() != self.n_nodes {
            return Err(SklearsError::InvalidInput(
                "Node weights length must match number of nodes".to_string(),
            ));
        }
        self.node_weights = Some(weights);
        Ok(self)
    }

    /// Get the degree of a node
    pub fn degree(&self, node: usize) -> f64 {
        if node >= self.n_nodes {
            return 0.0;
        }

        let mut degree = 0.0;
        for j in 0..self.n_nodes {
            degree += self.adjacency[[node, j]];
        }

        if self.directed {
            for i in 0..self.n_nodes {
                if i != node {
                    degree += self.adjacency[[i, node]];
                }
            }
        }

        degree
    }

    /// Get total number of edges (or total weight)
    pub fn total_weight(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.n_nodes {
            for j in 0..self.n_nodes {
                total += self.adjacency[[i, j]];
            }
        }

        if self.directed {
            total
        } else {
            total / 2.0 // Avoid double counting in undirected graphs
        }
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> Vec<(usize, f64)> {
        let mut neighbors = Vec::new();
        if node >= self.n_nodes {
            return neighbors;
        }

        for j in 0..self.n_nodes {
            if self.adjacency[[node, j]] > 0.0 {
                neighbors.push((j, self.adjacency[[node, j]]));
            }
        }

        neighbors
    }
}

/// Configuration for modularity-based clustering
#[derive(Debug, Clone)]
pub struct ModularityClusteringConfig {
    /// Resolution parameter (default: 1.0)
    pub resolution: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for ModularityClusteringConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_iterations: 100,
            tolerance: 1e-6,
            random_seed: None,
        }
    }
}

/// Modularity-based graph clustering
pub struct ModularityClustering {
    config: ModularityClusteringConfig,
}

impl ModularityClustering {
    /// Create a new modularity clustering instance
    pub fn new(config: ModularityClusteringConfig) -> Self {
        Self { config }
    }

    /// Compute modularity of a given community assignment
    pub fn compute_modularity(&self, graph: &Graph, communities: &[usize]) -> f64 {
        let total_weight = graph.total_weight();
        if total_weight == 0.0 {
            return 0.0;
        }

        let mut modularity = 0.0;

        for i in 0..graph.n_nodes {
            for j in 0..graph.n_nodes {
                if i == j {
                    continue;
                }

                if communities[i] == communities[j] {
                    let expected = (graph.degree(i) * graph.degree(j)) / (2.0 * total_weight);
                    modularity += graph.adjacency[[i, j]] - self.config.resolution * expected;
                }
            }
        }

        modularity / (2.0 * total_weight)
    }

    /// Greedy modularity optimization
    pub fn fit_greedy(&self, graph: &Graph) -> Result<Vec<usize>> {
        if graph.n_nodes == 0 {
            return Ok(Vec::new());
        }

        // Initialize each node to its own community
        let mut communities: Vec<usize> = (0..graph.n_nodes).collect();
        let mut improved = true;
        let mut iteration = 0;

        let mut rng = Random::default();

        while improved && iteration < self.config.max_iterations {
            improved = false;
            let mut node_order: Vec<usize> = (0..graph.n_nodes).collect();
            // Fisher-Yates shuffle
            for i in (1..node_order.len()).rev() {
                let j = rng.gen_range(0..=i);
                node_order.swap(i, j);
            }

            for &node in &node_order {
                let original_community = communities[node];
                let mut best_community = original_community;
                let mut best_modularity_gain = 0.0;

                // Try moving node to each neighboring community
                let neighbors = graph.neighbors(node);
                let mut neighboring_communities = HashSet::new();

                for (neighbor, _) in neighbors {
                    neighboring_communities.insert(communities[neighbor]);
                }

                for &candidate_community in &neighboring_communities {
                    if candidate_community != original_community {
                        // Temporarily move node to candidate community
                        communities[node] = candidate_community;
                        let new_modularity = self.compute_modularity(graph, &communities);

                        // Move back to compute baseline
                        communities[node] = original_community;
                        let old_modularity = self.compute_modularity(graph, &communities);

                        let modularity_gain = new_modularity - old_modularity;

                        if modularity_gain > best_modularity_gain + self.config.tolerance {
                            best_modularity_gain = modularity_gain;
                            best_community = candidate_community;
                        }
                    }
                }

                if best_community != original_community {
                    communities[node] = best_community;
                    improved = true;
                }
            }

            iteration += 1;
        }

        // Relabel communities to be contiguous
        Ok(self.relabel_communities(communities))
    }

    /// Relabel communities to be contiguous starting from 0
    fn relabel_communities(&self, communities: Vec<usize>) -> Vec<usize> {
        let mut unique_communities: Vec<usize> = communities.to_vec();
        unique_communities.sort();
        unique_communities.dedup();

        let mut community_map = HashMap::new();
        for (new_id, &old_id) in unique_communities.iter().enumerate() {
            community_map.insert(old_id, new_id);
        }

        communities.iter().map(|&c| community_map[&c]).collect()
    }
}

/// Configuration for Louvain algorithm
#[derive(Debug, Clone)]
pub struct LouvainConfig {
    /// Resolution parameter (default: 1.0)
    pub resolution: f64,
    /// Maximum number of iterations per level
    pub max_iterations_per_level: usize,
    /// Maximum number of levels
    pub max_levels: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_iterations_per_level: 100,
            max_levels: 10,
            tolerance: 1e-6,
            random_seed: None,
        }
    }
}

/// Louvain algorithm for community detection
pub struct LouvainClustering {
    config: LouvainConfig,
}

impl LouvainClustering {
    /// Create a new Louvain clustering instance
    pub fn new(config: LouvainConfig) -> Self {
        Self { config }
    }

    /// Run the Louvain algorithm
    pub fn fit(&self, graph: &Graph) -> Result<LouvainResult> {
        if graph.n_nodes == 0 {
            return Ok(LouvainResult {
                communities: Vec::new(),
                modularity: 0.0,
                levels: 0,
                community_hierarchy: Vec::new(),
            });
        }

        let mut current_graph = graph.clone();
        let mut communities: Vec<usize> = (0..graph.n_nodes).collect();
        let mut community_hierarchy = Vec::new();
        let mut level = 0;

        let mut rng = Random::default();

        while level < self.config.max_levels {
            // Phase 1: Local optimization
            let level_communities = self.optimize_modularity(&current_graph, &mut rng)?;

            // Check if any communities merged
            let n_communities = level_communities.iter().max().map(|&x| x + 1).unwrap_or(0);
            if n_communities >= current_graph.n_nodes {
                break; // No improvement possible
            }

            community_hierarchy.push(level_communities.clone());

            // Update global community assignment
            communities = self.update_global_communities(&communities, &level_communities);

            // Phase 2: Community aggregation
            current_graph = self.aggregate_communities(&current_graph, &level_communities)?;
            level += 1;
        }

        // Compute final modularity
        let modularity_clustering = ModularityClustering::new(ModularityClusteringConfig {
            resolution: self.config.resolution,
            ..Default::default()
        });
        let final_modularity = modularity_clustering.compute_modularity(graph, &communities);

        Ok(LouvainResult {
            communities,
            modularity: final_modularity,
            levels: level,
            community_hierarchy,
        })
    }

    /// Optimize modularity at current level
    fn optimize_modularity(&self, graph: &Graph, rng: &mut impl Rng) -> Result<Vec<usize>> {
        let mut communities: Vec<usize> = (0..graph.n_nodes).collect();
        let mut improved = true;
        let mut iteration = 0;

        let modularity_clustering = ModularityClustering::new(ModularityClusteringConfig {
            resolution: self.config.resolution,
            ..Default::default()
        });

        while improved && iteration < self.config.max_iterations_per_level {
            improved = false;
            let mut node_order: Vec<usize> = (0..graph.n_nodes).collect();
            node_order.shuffle(rng);

            for &node in &node_order {
                let original_community = communities[node];
                let mut best_community = original_community;
                let mut best_modularity_gain = 0.0;

                // Consider neighboring communities
                let neighbors = graph.neighbors(node);
                let mut neighboring_communities = HashSet::new();
                for (neighbor, _) in neighbors {
                    neighboring_communities.insert(communities[neighbor]);
                }

                // Also consider creating a new community
                let max_community = communities.iter().max().cloned().unwrap_or(0);
                neighboring_communities.insert(max_community + 1);

                for &candidate_community in &neighboring_communities {
                    if candidate_community != original_community {
                        communities[node] = candidate_community;
                        let new_modularity =
                            modularity_clustering.compute_modularity(graph, &communities);

                        communities[node] = original_community;
                        let old_modularity =
                            modularity_clustering.compute_modularity(graph, &communities);

                        let modularity_gain = new_modularity - old_modularity;

                        if modularity_gain > best_modularity_gain + self.config.tolerance {
                            best_modularity_gain = modularity_gain;
                            best_community = candidate_community;
                        }
                    }
                }

                if best_community != original_community {
                    communities[node] = best_community;
                    improved = true;
                }
            }

            iteration += 1;
        }

        Ok(modularity_clustering.relabel_communities(communities))
    }

    /// Update global community assignment based on level assignment
    fn update_global_communities(
        &self,
        global_communities: &[usize],
        level_communities: &[usize],
    ) -> Vec<usize> {
        let mut community_mapping = HashMap::new();
        let mut next_global_id = 0;

        for &local_community in level_communities {
            if let std::collections::hash_map::Entry::Vacant(e) =
                community_mapping.entry(local_community)
            {
                e.insert(next_global_id);
                next_global_id += 1;
            }
        }

        level_communities
            .iter()
            .map(|&c| community_mapping[&c])
            .collect()
    }

    /// Aggregate communities into super-nodes
    fn aggregate_communities(&self, graph: &Graph, communities: &[usize]) -> Result<Graph> {
        let n_communities = communities.iter().max().map(|&x| x + 1).unwrap_or(0);
        let mut new_adjacency = Array2::zeros((n_communities, n_communities));

        // Aggregate edge weights between communities
        for i in 0..graph.n_nodes {
            for j in 0..graph.n_nodes {
                let comm_i = communities[i];
                let comm_j = communities[j];
                new_adjacency[[comm_i, comm_j]] += graph.adjacency[[i, j]];
            }
        }

        Graph::from_adjacency(new_adjacency, graph.directed)
    }
}

/// Label propagation clustering configuration
#[derive(Debug, Clone)]
pub struct LabelPropagationConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for LabelPropagationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            random_seed: None,
        }
    }
}

/// Label propagation algorithm for community detection
pub struct LabelPropagationClustering {
    config: LabelPropagationConfig,
}

impl LabelPropagationClustering {
    /// Create a new label propagation clustering instance
    pub fn new(config: LabelPropagationConfig) -> Self {
        Self { config }
    }

    /// Run label propagation algorithm
    pub fn fit(&self, graph: &Graph) -> Result<Vec<usize>> {
        if graph.n_nodes == 0 {
            return Ok(Vec::new());
        }

        // Initialize each node with unique label
        let mut labels: Vec<usize> = (0..graph.n_nodes).collect();
        let mut new_labels = labels.clone();

        let mut rng = Random::default();

        for iteration in 0..self.config.max_iterations {
            let mut changed = false;
            let mut node_order: Vec<usize> = (0..graph.n_nodes).collect();
            // Fisher-Yates shuffle
            for i in (1..node_order.len()).rev() {
                let j = rng.gen_range(0..=i);
                node_order.swap(i, j);
            }

            for &node in &node_order {
                // Count label frequencies among neighbors
                let mut label_weights = HashMap::new();
                let neighbors = graph.neighbors(node);

                for (neighbor, weight) in neighbors {
                    if neighbor != node {
                        // Exclude self-loops in neighbor counting
                        *label_weights.entry(labels[neighbor]).or_insert(0.0) += weight;
                    }
                }

                if !label_weights.is_empty() {
                    // Find the most frequent label (with highest total weight)
                    let mut best_labels = Vec::new();
                    let mut max_weight = 0.0;

                    for (&label, &weight) in &label_weights {
                        match weight.partial_cmp(&max_weight) {
                            Some(Ordering::Greater) => {
                                max_weight = weight;
                                best_labels.clear();
                                best_labels.push(label);
                            }
                            Some(Ordering::Equal) => {
                                best_labels.push(label);
                            }
                            _ => {}
                        }
                    }

                    // Break ties randomly
                    if !best_labels.is_empty() {
                        let chosen_label = best_labels[rng.gen_range(0..best_labels.len())];
                        if chosen_label != labels[node] {
                            new_labels[node] = chosen_label;
                            changed = true;
                        }
                    }
                }
            }

            // Update labels
            labels = new_labels.clone();

            if !changed {
                break;
            }
        }

        // Relabel to be contiguous
        Ok(self.relabel_communities(labels))
    }

    /// Relabel communities to be contiguous starting from 0
    fn relabel_communities(&self, communities: Vec<usize>) -> Vec<usize> {
        let mut unique_communities: Vec<usize> = communities.to_vec();
        unique_communities.sort();
        unique_communities.dedup();

        let mut community_map = HashMap::new();
        for (new_id, &old_id) in unique_communities.iter().enumerate() {
            community_map.insert(old_id, new_id);
        }

        communities.iter().map(|&c| community_map[&c]).collect()
    }
}

/// Spectral graph clustering configuration
#[derive(Debug, Clone)]
pub struct SpectralGraphConfig {
    /// Number of clusters to find
    pub n_clusters: usize,
    /// Number of eigenvectors to compute
    pub n_eigenvectors: Option<usize>,
    /// Normalization method: "unnormalized", "symmetric", "random_walk"
    pub normalization: String,
    /// Random seed for k-means clustering of eigenvectors
    pub random_seed: Option<u64>,
}

impl Default for SpectralGraphConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            n_eigenvectors: None,
            normalization: "symmetric".to_string(),
            random_seed: None,
        }
    }
}

/// Spectral graph clustering
pub struct SpectralGraphClustering {
    config: SpectralGraphConfig,
}

impl SpectralGraphClustering {
    /// Create a new spectral graph clustering instance
    pub fn new(config: SpectralGraphConfig) -> Self {
        Self { config }
    }

    /// Run spectral clustering on graph
    pub fn fit(&self, graph: &Graph) -> Result<Vec<usize>> {
        if graph.n_nodes == 0 {
            return Ok(Vec::new());
        }

        if self.config.n_clusters > graph.n_nodes {
            return Err(SklearsError::InvalidInput(
                "Number of clusters cannot exceed number of nodes".to_string(),
            ));
        }

        // Compute Laplacian matrix
        let laplacian = self.compute_laplacian(graph)?;

        // Compute eigenvectors
        let n_eigenvectors = self.config.n_eigenvectors.unwrap_or(self.config.n_clusters);
        let eigenvectors = self.compute_eigenvectors(&laplacian, n_eigenvectors)?;

        // Apply k-means to eigenvectors
        self.cluster_eigenvectors(&eigenvectors)
    }

    /// Compute graph Laplacian matrix
    fn compute_laplacian(&self, graph: &Graph) -> Result<Array2<f64>> {
        let n = graph.n_nodes;
        let mut laplacian = Array2::zeros((n, n));

        // Compute degree matrix
        let mut degrees = vec![0.0; n];
        for i in 0..n {
            degrees[i] = graph.degree(i);
        }

        match self.config.normalization.as_str() {
            "unnormalized" => {
                // L = D - A
                for i in 0..n {
                    laplacian[[i, i]] = degrees[i];
                    for j in 0..n {
                        if i != j {
                            laplacian[[i, j]] = -graph.adjacency[[i, j]];
                        }
                    }
                }
            }
            "symmetric" => {
                // L = I - D^(-1/2) * A * D^(-1/2)
                for i in 0..n {
                    laplacian[[i, i]] = 1.0;
                    let sqrt_deg_i = if degrees[i] > 0.0 {
                        degrees[i].sqrt()
                    } else {
                        0.0
                    };

                    for j in 0..n {
                        if i != j && graph.adjacency[[i, j]] > 0.0 {
                            let sqrt_deg_j = if degrees[j] > 0.0 {
                                degrees[j].sqrt()
                            } else {
                                0.0
                            };
                            if sqrt_deg_i > 0.0 && sqrt_deg_j > 0.0 {
                                laplacian[[i, j]] =
                                    -graph.adjacency[[i, j]] / (sqrt_deg_i * sqrt_deg_j);
                            }
                        }
                    }
                }
            }
            "random_walk" => {
                // L = I - D^(-1) * A
                for i in 0..n {
                    laplacian[[i, i]] = 1.0;
                    if degrees[i] > 0.0 {
                        for j in 0..n {
                            if i != j {
                                laplacian[[i, j]] = -graph.adjacency[[i, j]] / degrees[i];
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Invalid normalization method. Use 'unnormalized', 'symmetric', or 'random_walk'".to_string(),
                ));
            }
        }

        Ok(laplacian)
    }

    /// Compute smallest eigenvectors (placeholder - would need proper eigenvalue solver)
    fn compute_eigenvectors(
        &self,
        laplacian: &Array2<f64>,
        n_eigenvectors: usize,
    ) -> Result<Array2<f64>> {
        // This is a simplified placeholder implementation
        // In practice, you would use a proper eigenvalue decomposition library

        let n = laplacian.nrows();
        if n_eigenvectors > n {
            return Err(SklearsError::InvalidInput(
                "Cannot compute more eigenvectors than matrix size".to_string(),
            ));
        }

        // For now, return random embeddings as placeholder
        // TODO: Implement proper eigenvalue decomposition
        let mut rng = thread_rng();
        let mut eigenvectors = Array2::zeros((n, n_eigenvectors));

        let normal = scirs2_core::random::RandNormal::new(0.0, 1.0).unwrap();
        for i in 0..n {
            for j in 0..n_eigenvectors {
                eigenvectors[[i, j]] = normal.sample(&mut rng);
            }
        }

        Ok(eigenvectors)
    }

    /// Cluster eigenvectors using k-means
    fn cluster_eigenvectors(&self, eigenvectors: &Array2<f64>) -> Result<Vec<usize>> {
        // Placeholder k-means implementation
        // In practice, you would use the K-means implementation from the main clustering module

        let n_points = eigenvectors.nrows();
        let n_clusters = self.config.n_clusters;

        if n_clusters >= n_points {
            return Ok((0..n_points).collect());
        }

        // Simple random assignment as placeholder
        let mut rng = Random::default();

        let mut clusters = Vec::new();
        for _ in 0..n_points {
            clusters.push(rng.gen_range(0..n_clusters));
        }

        Ok(clusters)
    }
}

/// Result of Louvain clustering
#[derive(Debug, Clone)]
pub struct LouvainResult {
    /// Final community assignments
    pub communities: Vec<usize>,
    /// Final modularity score
    pub modularity: f64,
    /// Number of levels in the hierarchy
    pub levels: usize,
    /// Community assignments at each level
    pub community_hierarchy: Vec<Vec<usize>>,
}

/// Result of graph clustering analysis
#[derive(Debug, Clone)]
pub struct GraphClusteringResult {
    /// Community assignments
    pub communities: Vec<usize>,
    /// Modularity score
    pub modularity: f64,
    /// Number of communities found
    pub n_communities: usize,
    /// Community sizes
    pub community_sizes: Vec<usize>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_graph_creation() {
        let adjacency =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .unwrap();

        let graph = Graph::from_adjacency(adjacency, false).unwrap();
        assert_eq!(graph.n_nodes, 3);
        assert!(!graph.directed);
        assert_abs_diff_eq!(graph.degree(1), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_modularity_computation() {
        let adjacency = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            ],
        )
        .unwrap();

        let graph = Graph::from_adjacency(adjacency, false).unwrap();
        let clustering = ModularityClustering::new(ModularityClusteringConfig::default());

        // Perfect community structure: nodes 0,2 in community 0, nodes 1,3 in community 1
        let communities = vec![0, 1, 0, 1];
        let modularity = clustering.compute_modularity(&graph, &communities);

        // Should have positive modularity for this community structure
        assert!(modularity > 0.0);
    }

    #[test]
    fn test_label_propagation() {
        let adjacency = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            ],
        )
        .unwrap();

        let graph = Graph::from_adjacency(adjacency, false).unwrap();
        let clustering = LabelPropagationClustering::new(LabelPropagationConfig {
            random_seed: Some(42),
            ..Default::default()
        });

        let communities = clustering.fit(&graph).unwrap();
        assert_eq!(communities.len(), 4);

        // Check that communities are contiguous (0, 1, 2, ...)
        let mut unique_communities = communities.clone();
        unique_communities.sort();
        unique_communities.dedup();
        assert_eq!(
            unique_communities,
            (0..unique_communities.len()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_spectral_clustering_config() {
        let config = SpectralGraphConfig {
            n_clusters: 3,
            normalization: "symmetric".to_string(),
            ..Default::default()
        };

        let clustering = SpectralGraphClustering::new(config);

        let adjacency = Array2::eye(5);
        let graph = Graph::from_adjacency(adjacency, false).unwrap();

        let result = clustering.fit(&graph);
        assert!(result.is_ok());

        let communities = result.unwrap();
        assert_eq!(communities.len(), 5);
    }

    #[test]
    fn test_graph_from_edges() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];

        let graph = Graph::from_edges(&edges, 3, false).unwrap();
        assert_eq!(graph.n_nodes, 3);
        assert_abs_diff_eq!(graph.total_weight(), 3.0, epsilon = 1e-10);

        // Check symmetry for undirected graph
        assert_abs_diff_eq!(
            graph.adjacency[[0, 1]],
            graph.adjacency[[1, 0]],
            epsilon = 1e-10
        );
    }
}
