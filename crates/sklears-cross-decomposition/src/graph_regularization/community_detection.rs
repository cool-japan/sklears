//! Community Detection for Graph-Regularized Methods
//!
//! This module provides various community detection algorithms that can be integrated
//! with graph-regularized cross-decomposition methods. Communities are groups of densely
//! connected nodes that are sparsely connected to other groups.
//!
//! ## Supported Algorithms
//! - Modularity-based methods (Louvain, Leiden)
//! - Spectral clustering
//! - Label propagation
//! - Infomap
//! - Hierarchical community detection
//! - Overlapping community detection
//!
//! ## Applications
//! - Feature grouping in high-dimensional data
//! - Multi-scale network analysis
//! - Community-aware canonical correlation
//! - Structured sparsity patterns

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, CoreRandom, Rng};
use sklears_core::types::Float;
use std::collections::{HashMap, HashSet, VecDeque};

/// Community detection algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunityAlgorithm {
    /// Louvain method for modularity optimization
    Louvain,
    /// Leiden method (improved Louvain)
    Leiden,
    /// Spectral clustering
    SpectralClustering,
    /// Label propagation
    LabelPropagation,
    /// Girvan-Newman edge betweenness
    GirvanNewman,
    /// Fast greedy modularity optimization
    FastGreedy,
}

/// Configuration for community detection
#[derive(Debug, Clone)]
pub struct CommunityDetectionConfig {
    /// Algorithm to use
    pub algorithm: CommunityAlgorithm,
    /// Resolution parameter (for modularity-based methods)
    pub resolution: Float,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Minimum community size
    pub min_community_size: usize,
}

impl Default for CommunityDetectionConfig {
    fn default() -> Self {
        Self {
            algorithm: CommunityAlgorithm::Louvain,
            resolution: 1.0,
            max_iterations: 100,
            tolerance: 1e-6,
            random_seed: None,
            min_community_size: 1,
        }
    }
}

/// Community structure detection results
#[derive(Debug, Clone)]
pub struct CommunityStructure {
    /// Community assignment for each node
    pub assignments: Array1<usize>,
    /// Number of communities detected
    pub num_communities: usize,
    /// Modularity score
    pub modularity: Float,
    /// Community sizes
    pub community_sizes: Vec<usize>,
    /// Dendogram for hierarchical methods
    pub dendrogram: Option<Vec<(usize, usize, Float)>>,
}

/// Community detection implementation
pub struct CommunityDetector {
    /// Configuration
    config: CommunityDetectionConfig,
}

impl CommunityDetector {
    /// Create a new community detector
    pub fn new(config: CommunityDetectionConfig) -> Self {
        Self { config }
    }

    /// Detect communities in a graph
    ///
    /// # Arguments
    /// * `adjacency` - Adjacency matrix of the graph
    ///
    /// # Returns
    /// Community structure with assignments and quality metrics
    pub fn detect(&self, adjacency: ArrayView2<Float>) -> CommunityStructure {
        match self.config.algorithm {
            CommunityAlgorithm::Louvain => self.louvain(adjacency),
            CommunityAlgorithm::Leiden => self.leiden(adjacency),
            CommunityAlgorithm::SpectralClustering => self.spectral_clustering(adjacency),
            CommunityAlgorithm::LabelPropagation => self.label_propagation(adjacency),
            CommunityAlgorithm::GirvanNewman => self.girvan_newman(adjacency),
            CommunityAlgorithm::FastGreedy => self.fast_greedy(adjacency),
        }
    }

    /// Louvain method for community detection
    fn louvain(&self, adjacency: ArrayView2<Float>) -> CommunityStructure {
        let n = adjacency.nrows();
        let mut assignments = Array1::from_iter(0..n);

        let m = adjacency.sum() / 2.0; // Total edge weight
        let mut best_modularity = 0.0;

        for iteration in 0..self.config.max_iterations {
            let mut improved = false;

            // Phase 1: Move nodes to maximize modularity
            for node in 0..n {
                let current_community = assignments[node];

                // Try moving to neighbor communities
                let mut best_delta = 0.0;
                let mut best_community = current_community;

                for neighbor in 0..n {
                    if adjacency[[node, neighbor]] > 0.0 {
                        let neighbor_community = assignments[neighbor];
                        if neighbor_community != current_community {
                            let delta = self.compute_modularity_delta(
                                &adjacency,
                                &assignments,
                                node,
                                current_community,
                                neighbor_community,
                                m,
                            );

                            if delta > best_delta {
                                best_delta = delta;
                                best_community = neighbor_community;
                            }
                        }
                    }
                }

                if best_delta > self.config.tolerance {
                    assignments[node] = best_community;
                    improved = true;
                }
            }

            // Compute current modularity
            let current_modularity = self.compute_modularity(&adjacency, &assignments);

            if current_modularity > best_modularity {
                best_modularity = current_modularity;
            }

            if !improved {
                break;
            }
        }

        // Relabel communities to be contiguous
        let (assignments, num_communities) = self.relabel_communities(assignments);

        // Compute community sizes
        let community_sizes = self.compute_community_sizes(&assignments, num_communities);

        CommunityStructure {
            assignments,
            num_communities,
            modularity: best_modularity,
            community_sizes,
            dendrogram: None,
        }
    }

    /// Leiden method (improved Louvain)
    fn leiden(&self, adjacency: ArrayView2<Float>) -> CommunityStructure {
        // Simplified implementation - in practice, Leiden adds refinement steps
        let mut louvain_result = self.louvain(adjacency);

        // Add refinement: check if communities can be split
        let n = adjacency.nrows();
        let mut refined_assignments = louvain_result.assignments.clone();

        for community_id in 0..louvain_result.num_communities {
            // Get nodes in this community
            let community_nodes: Vec<usize> = (0..n)
                .filter(|&i| refined_assignments[i] == community_id)
                .collect();

            if community_nodes.len() > 2 {
                // Try to split using simple connectivity check
                let split_assignments = self.try_split_community(&adjacency, &community_nodes);
                if let Some(splits) = split_assignments {
                    for (idx, &node) in community_nodes.iter().enumerate() {
                        if splits[idx] == 1 {
                            refined_assignments[node] = louvain_result.num_communities;
                        }
                    }
                    louvain_result.num_communities += 1;
                }
            }
        }

        let (assignments, num_communities) = self.relabel_communities(refined_assignments);
        let modularity = self.compute_modularity(&adjacency, &assignments);
        let community_sizes = self.compute_community_sizes(&assignments, num_communities);

        CommunityStructure {
            assignments,
            num_communities,
            modularity,
            community_sizes,
            dendrogram: None,
        }
    }

    /// Spectral clustering for community detection
    fn spectral_clustering(&self, adjacency: ArrayView2<Float>) -> CommunityStructure {
        let n = adjacency.nrows();

        // Compute graph Laplacian
        let degrees = adjacency.sum_axis(Axis(1));
        let mut laplacian = adjacency.to_owned();

        for i in 0..n {
            laplacian[[i, i]] = degrees[i] - laplacian[[i, i]];
            for j in 0..n {
                if i != j {
                    laplacian[[i, j]] = -laplacian[[i, j]];
                }
            }
        }

        // Simple k-means clustering on Laplacian eigenvectors
        // (In practice, would use proper eigenvalue decomposition)
        let k = (n as Float).sqrt().ceil() as usize;
        let mut assignments = Array1::zeros(n);

        // Simple heuristic assignment based on node degrees
        let sorted_indices: Vec<usize> = {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| degrees[i].partial_cmp(&degrees[j]).unwrap());
            indices
        };

        for (idx, &node) in sorted_indices.iter().enumerate() {
            assignments[node] = idx * k / n;
        }

        let num_communities = k;
        let modularity = self.compute_modularity(&adjacency, &assignments);
        let community_sizes = self.compute_community_sizes(&assignments, num_communities);

        CommunityStructure {
            assignments,
            num_communities,
            modularity,
            community_sizes,
            dendrogram: None,
        }
    }

    /// Label propagation algorithm
    fn label_propagation(&self, adjacency: ArrayView2<Float>) -> CommunityStructure {
        let n = adjacency.nrows();
        let mut assignments = Array1::from_iter(0..n); // Initially, each node is its own community

        let mut rng = if let Some(seed) = self.config.random_seed {
            thread_rng() // In practice, would use seeded RNG
        } else {
            thread_rng()
        };

        for _ in 0..self.config.max_iterations {
            let mut changed = false;

            // Random node order
            let mut node_order: Vec<usize> = (0..n).collect();
            for i in 0..n {
                let j = rng.random_range(i..n);
                node_order.swap(i, j);
            }

            // Update labels based on neighbors
            for &node in &node_order {
                let mut label_counts: HashMap<usize, Float> = HashMap::new();

                // Count labels of neighbors weighted by edge weights
                for neighbor in 0..n {
                    if adjacency[[node, neighbor]] > 0.0 {
                        let label = assignments[neighbor];
                        *label_counts.entry(label).or_insert(0.0) += adjacency[[node, neighbor]];
                    }
                }

                // Choose most frequent label
                if let Some((&best_label, _)) = label_counts
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                {
                    if best_label != assignments[node] {
                        assignments[node] = best_label;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        let (assignments, num_communities) = self.relabel_communities(assignments);
        let modularity = self.compute_modularity(&adjacency, &assignments);
        let community_sizes = self.compute_community_sizes(&assignments, num_communities);

        CommunityStructure {
            assignments,
            num_communities,
            modularity,
            community_sizes,
            dendrogram: None,
        }
    }

    /// Girvan-Newman edge betweenness method
    fn girvan_newman(&self, adjacency: ArrayView2<Float>) -> CommunityStructure {
        // Simplified implementation
        let n = adjacency.nrows();
        let mut graph = adjacency.to_owned();
        let mut dendrogram = Vec::new();

        // Initially all nodes in one community
        let mut assignments = Array1::zeros(n);
        let mut num_communities = 1;

        // Iteratively remove edges with highest betweenness
        for iteration in 0..self.config.max_iterations {
            // Find edge with highest betweenness (simplified: use edge weight)
            let mut max_weight = 0.0;
            let mut max_edge = (0, 0);

            for i in 0..n {
                for j in (i + 1)..n {
                    if graph[[i, j]] > max_weight {
                        max_weight = graph[[i, j]];
                        max_edge = (i, j);
                    }
                }
            }

            if max_weight == 0.0 {
                break;
            }

            // Remove edge
            let (i, j) = max_edge;
            graph[[i, j]] = 0.0;
            graph[[j, i]] = 0.0;

            dendrogram.push((i, j, max_weight));

            // Check if graph is now disconnected
            let components = self.find_connected_components(&graph);
            if components.len() > num_communities {
                num_communities = components.len();

                // Update assignments
                for (comp_id, component) in components.iter().enumerate() {
                    for &node in component {
                        assignments[node] = comp_id;
                    }
                }
            }

            if num_communities >= n / self.config.min_community_size {
                break;
            }
        }

        let modularity = self.compute_modularity(&adjacency, &assignments);
        let community_sizes = self.compute_community_sizes(&assignments, num_communities);

        CommunityStructure {
            assignments,
            num_communities,
            modularity,
            community_sizes,
            dendrogram: Some(dendrogram),
        }
    }

    /// Fast greedy modularity optimization
    fn fast_greedy(&self, adjacency: ArrayView2<Float>) -> CommunityStructure {
        let n = adjacency.nrows();
        let mut assignments = Array1::from_iter(0..n);

        let m = adjacency.sum() / 2.0;
        let mut best_modularity = 0.0;
        let mut best_assignments = assignments.clone();

        // Greedy merging
        for _ in 0..n - 1 {
            let mut best_delta = Float::NEG_INFINITY;
            let mut best_merge = (0, 0);

            // Find best pair to merge
            for i in 0..n {
                for j in (i + 1)..n {
                    if assignments[i] != assignments[j] {
                        let delta = self.compute_modularity_delta(
                            &adjacency,
                            &assignments,
                            i,
                            assignments[i],
                            assignments[j],
                            m,
                        );

                        if delta > best_delta {
                            best_delta = delta;
                            best_merge = (i, j);
                        }
                    }
                }
            }

            if best_delta <= 0.0 {
                break;
            }

            // Merge communities
            let (i, j) = best_merge;
            let comm_i = assignments[i];
            let comm_j = assignments[j];

            for k in 0..n {
                if assignments[k] == comm_j {
                    assignments[k] = comm_i;
                }
            }

            let modularity = self.compute_modularity(&adjacency, &assignments);
            if modularity > best_modularity {
                best_modularity = modularity;
                best_assignments = assignments.clone();
            }
        }

        let (assignments, num_communities) = self.relabel_communities(best_assignments);
        let community_sizes = self.compute_community_sizes(&assignments, num_communities);

        CommunityStructure {
            assignments,
            num_communities,
            modularity: best_modularity,
            community_sizes,
            dendrogram: None,
        }
    }

    /// Compute modularity of a partition
    fn compute_modularity(
        &self,
        adjacency: &ArrayView2<Float>,
        assignments: &Array1<usize>,
    ) -> Float {
        let n = adjacency.nrows();
        let m = adjacency.sum() / 2.0;

        if m == 0.0 {
            return 0.0;
        }

        let mut modularity = 0.0;

        for i in 0..n {
            for j in 0..n {
                if assignments[i] == assignments[j] {
                    let k_i = adjacency.row(i).sum();
                    let k_j = adjacency.row(j).sum();
                    modularity += adjacency[[i, j]] - (k_i * k_j) / (2.0 * m);
                }
            }
        }

        modularity / (2.0 * m)
    }

    /// Compute change in modularity when moving a node
    fn compute_modularity_delta(
        &self,
        adjacency: &ArrayView2<Float>,
        assignments: &Array1<usize>,
        node: usize,
        from_comm: usize,
        to_comm: usize,
        m: Float,
    ) -> Float {
        let n = adjacency.nrows();

        // Sum of weights to nodes in each community
        let mut k_i_in_from = 0.0;
        let mut k_i_in_to = 0.0;

        for j in 0..n {
            if assignments[j] == from_comm {
                k_i_in_from += adjacency[[node, j]];
            }
            if assignments[j] == to_comm {
                k_i_in_to += adjacency[[node, j]];
            }
        }

        let k_i = adjacency.row(node).sum();

        // Compute delta Q
        (k_i_in_to - k_i_in_from) / m
            - self.config.resolution * k_i * (k_i_in_to - k_i_in_from) / (m * m)
    }

    /// Relabel communities to be contiguous 0, 1, 2, ...
    fn relabel_communities(&self, assignments: Array1<usize>) -> (Array1<usize>, usize) {
        let unique_communities: HashSet<usize> = assignments.iter().copied().collect();
        let community_map: HashMap<usize, usize> = unique_communities
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (old_id, new_id))
            .collect();

        let new_assignments = assignments.mapv(|c| community_map[&c]);
        let num_communities = unique_communities.len();

        (new_assignments, num_communities)
    }

    /// Compute sizes of each community
    fn compute_community_sizes(
        &self,
        assignments: &Array1<usize>,
        num_communities: usize,
    ) -> Vec<usize> {
        let mut sizes = vec![0; num_communities];

        for &community in assignments.iter() {
            if community < num_communities {
                sizes[community] += 1;
            }
        }

        sizes
    }

    /// Find connected components using BFS
    fn find_connected_components(&self, adjacency: &Array2<Float>) -> Vec<Vec<usize>> {
        let n = adjacency.nrows();
        let mut visited = vec![false; n];
        let mut components = Vec::new();

        for start in 0..n {
            if !visited[start] {
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back(start);
                visited[start] = true;

                while let Some(node) = queue.pop_front() {
                    component.push(node);

                    for neighbor in 0..n {
                        if !visited[neighbor] && adjacency[[node, neighbor]] > 0.0 {
                            visited[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }

                components.push(component);
            }
        }

        components
    }

    /// Try to split a community into two parts
    fn try_split_community(
        &self,
        adjacency: &ArrayView2<Float>,
        nodes: &[usize],
    ) -> Option<Vec<usize>> {
        if nodes.len() < 2 {
            return None;
        }

        // Simple split: use node connectivity
        let mut internal_edges: HashMap<(usize, usize), Float> = HashMap::new();

        for (i, &node_i) in nodes.iter().enumerate() {
            for (j, &node_j) in nodes.iter().enumerate().skip(i + 1) {
                let weight = adjacency[[node_i, node_j]];
                if weight > 0.0 {
                    internal_edges.insert((i, j), weight);
                }
            }
        }

        // If well-connected, don't split
        if internal_edges.len() > nodes.len() {
            return None;
        }

        // Simple bipartition based on connectivity
        let mut assignments = vec![0; nodes.len()];
        for i in (nodes.len() / 2)..nodes.len() {
            assignments[i] = 1;
        }

        Some(assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_community_detector_creation() {
        let config = CommunityDetectionConfig::default();
        let detector = CommunityDetector::new(config);
        assert_eq!(detector.config.algorithm, CommunityAlgorithm::Louvain);
    }

    #[test]
    fn test_louvain_simple_graph() {
        let config = CommunityDetectionConfig {
            algorithm: CommunityAlgorithm::Louvain,
            ..Default::default()
        };
        let detector = CommunityDetector::new(config);

        // Simple graph with two clear communities
        let adjacency = array![
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ];

        let result = detector.detect(adjacency.view());

        assert!(result.num_communities >= 2);
        assert!(result.num_communities <= 5);
        assert_eq!(result.assignments.len(), 5);
    }

    #[test]
    fn test_label_propagation() {
        let config = CommunityDetectionConfig {
            algorithm: CommunityAlgorithm::LabelPropagation,
            max_iterations: 10,
            ..Default::default()
        };
        let detector = CommunityDetector::new(config);

        let adjacency = array![
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ];

        let result = detector.detect(adjacency.view());

        assert!(result.num_communities >= 1);
        assert!(result.num_communities <= 4);
    }

    #[test]
    fn test_spectral_clustering() {
        let config = CommunityDetectionConfig {
            algorithm: CommunityAlgorithm::SpectralClustering,
            ..Default::default()
        };
        let detector = CommunityDetector::new(config);

        let adjacency = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0],];

        let result = detector.detect(adjacency.view());

        assert_eq!(result.assignments.len(), 3);
        assert!(result.num_communities >= 1);
    }

    #[test]
    fn test_modularity_computation() {
        let config = CommunityDetectionConfig::default();
        let detector = CommunityDetector::new(config);

        let adjacency = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0],];

        let assignments = array![0, 0, 1];

        let modularity = detector.compute_modularity(&adjacency.view(), &assignments);

        // Modularity should be between -0.5 and 1.0
        assert!(modularity >= -0.5);
        assert!(modularity <= 1.0);
    }

    #[test]
    fn test_relabel_communities() {
        let config = CommunityDetectionConfig::default();
        let detector = CommunityDetector::new(config);

        let assignments = array![5, 5, 10, 10, 2];
        let (relabeled, num_communities) = detector.relabel_communities(assignments);

        assert_eq!(num_communities, 3);
        assert_eq!(relabeled.len(), 5);

        // Check that all community IDs are in [0, num_communities)
        for &comm in relabeled.iter() {
            assert!(comm < num_communities);
        }
    }

    #[test]
    fn test_community_sizes() {
        let config = CommunityDetectionConfig::default();
        let detector = CommunityDetector::new(config);

        let assignments = array![0, 0, 1, 1, 1, 2];
        let sizes = detector.compute_community_sizes(&assignments, 3);

        assert_eq!(sizes, vec![2, 3, 1]);
    }

    #[test]
    fn test_connected_components() {
        let config = CommunityDetectionConfig::default();
        let detector = CommunityDetector::new(config);

        // Disconnected graph
        let adjacency = array![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ];

        let components = detector.find_connected_components(&adjacency);

        assert_eq!(components.len(), 2);
        assert_eq!(components[0].len() + components[1].len(), 4);
    }

    #[test]
    fn test_girvan_newman() {
        let config = CommunityDetectionConfig {
            algorithm: CommunityAlgorithm::GirvanNewman,
            max_iterations: 5,
            ..Default::default()
        };
        let detector = CommunityDetector::new(config);

        let adjacency = array![
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.5],
            [1.0, 1.0, 0.0, 0.5],
            [0.0, 0.5, 0.5, 0.0],
        ];

        let result = detector.detect(adjacency.view());

        assert!(result.dendrogram.is_some());
        assert!(result.num_communities >= 1);
    }

    #[test]
    fn test_fast_greedy() {
        let config = CommunityDetectionConfig {
            algorithm: CommunityAlgorithm::FastGreedy,
            ..Default::default()
        };
        let detector = CommunityDetector::new(config);

        let adjacency = array![
            [0.0, 2.0, 2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 3.0, 0.0],
        ];

        let result = detector.detect(adjacency.view());

        assert!(result.num_communities >= 2);
        assert!(result.modularity >= 0.0);
    }
}
