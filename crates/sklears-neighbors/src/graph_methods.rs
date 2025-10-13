//! Graph-based neighbor methods for machine learning
//!
//! This module implements various graph-based neighbor algorithms including
//! k-nearest neighbor graphs, mutual k-nearest neighbors, epsilon graphs,
//! relative neighborhood graphs, and Gabriel graphs.

use crate::distance::Distance;
use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::types::Float;
use std::collections::{HashMap, HashSet};

/// Represents a graph edge with weight
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub source: usize,
    pub target: usize,
    pub weight: Float,
}

/// Represents a neighborhood graph
#[derive(Debug, Clone)]
pub struct NeighborhoodGraph {
    pub nodes: usize,
    pub edges: Vec<GraphEdge>,
    pub adjacency_matrix: Array2<Float>,
}

impl NeighborhoodGraph {
    /// Create a new neighborhood graph with given number of nodes
    pub fn new(nodes: usize) -> Self {
        Self {
            nodes,
            edges: Vec::new(),
            adjacency_matrix: Array2::zeros((nodes, nodes)),
        }
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, source: usize, target: usize, weight: Float) {
        if source < self.nodes && target < self.nodes {
            self.edges.push(GraphEdge {
                source,
                target,
                weight,
            });
            self.adjacency_matrix[[source, target]] = weight;
        }
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node: usize) -> Vec<usize> {
        if node >= self.nodes {
            return Vec::new();
        }

        let mut neighbors = Vec::new();
        for j in 0..self.nodes {
            if self.adjacency_matrix[[node, j]] > 0.0 {
                neighbors.push(j);
            }
        }
        neighbors
    }

    /// Get the degree of a node
    pub fn degree(&self, node: usize) -> usize {
        if node >= self.nodes {
            return 0;
        }

        self.adjacency_matrix
            .row(node)
            .iter()
            .filter(|&&x| x > 0.0)
            .count()
    }

    /// Check if there's an edge between two nodes
    pub fn has_edge(&self, source: usize, target: usize) -> bool {
        if source >= self.nodes || target >= self.nodes {
            return false;
        }
        self.adjacency_matrix[[source, target]] > 0.0
    }

    /// Get the weight of an edge between two nodes
    pub fn get_edge_weight(&self, source: usize, target: usize) -> Float {
        if source >= self.nodes || target >= self.nodes {
            return 0.0;
        }
        self.adjacency_matrix[[source, target]]
    }

    /// Convert to sparse representation (list of edges)
    pub fn to_edge_list(&self) -> Vec<(usize, usize, Float)> {
        self.edges
            .iter()
            .map(|e| (e.source, e.target, e.weight))
            .collect()
    }

    /// Get graph statistics
    pub fn statistics(&self) -> GraphStatistics {
        let total_edges = self.edges.len();
        let max_degree = (0..self.nodes).map(|i| self.degree(i)).max().unwrap_or(0);
        let min_degree = (0..self.nodes).map(|i| self.degree(i)).min().unwrap_or(0);
        let avg_degree = if self.nodes > 0 {
            total_edges as Float / self.nodes as Float
        } else {
            0.0
        };

        GraphStatistics {
            nodes: self.nodes,
            edges: total_edges,
            max_degree,
            min_degree,
            avg_degree,
        }
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    pub nodes: usize,
    pub edges: usize,
    pub max_degree: usize,
    pub min_degree: usize,
    pub avg_degree: Float,
}

/// K-Nearest Neighbor Graph Builder
pub struct KNearestNeighborGraph {
    k: usize,
    distance: Distance,
    directed: bool,
}

impl KNearestNeighborGraph {
    /// Create a new k-nearest neighbor graph builder
    pub fn new(k: usize) -> Self {
        Self {
            k,
            distance: Distance::Euclidean,
            directed: true,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set whether the graph should be directed
    pub fn with_directed(mut self, directed: bool) -> Self {
        self.directed = directed;
        self
    }

    /// Build the k-nearest neighbor graph
    pub fn build(&self, data: &ArrayView2<Float>) -> NeighborsResult<NeighborhoodGraph> {
        let n_samples = data.nrows();

        if n_samples == 0 {
            return Err(NeighborsError::EmptyInput);
        }

        if self.k >= n_samples {
            return Err(NeighborsError::InvalidNeighbors(self.k));
        }

        let mut graph = NeighborhoodGraph::new(n_samples);

        // For each sample, find k nearest neighbors
        for i in 0..n_samples {
            let mut distances: Vec<(usize, Float)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = self.distance.calculate(&data.row(i), &data.row(j));
                    distances.push((j, dist));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for &(neighbor, dist) in distances.iter().take(self.k) {
                graph.add_edge(i, neighbor, dist);

                // If undirected, add reverse edge
                if !self.directed {
                    graph.add_edge(neighbor, i, dist);
                }
            }
        }

        Ok(graph)
    }
}

/// Mutual K-Nearest Neighbors Graph Builder
pub struct MutualKNearestNeighbors {
    k: usize,
    distance: Distance,
}

impl MutualKNearestNeighbors {
    /// Create a new mutual k-nearest neighbors graph builder
    pub fn new(k: usize) -> Self {
        Self {
            k,
            distance: Distance::Euclidean,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Build the mutual k-nearest neighbors graph
    pub fn build(&self, data: &ArrayView2<Float>) -> NeighborsResult<NeighborhoodGraph> {
        let n_samples = data.nrows();

        if n_samples == 0 {
            return Err(NeighborsError::EmptyInput);
        }

        if self.k >= n_samples {
            return Err(NeighborsError::InvalidNeighbors(self.k));
        }

        let mut graph = NeighborhoodGraph::new(n_samples);
        let mut knn_sets: Vec<HashSet<usize>> = Vec::new();

        // First, compute k-nearest neighbors for each sample
        for i in 0..n_samples {
            let mut distances: Vec<(usize, Float)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = self.distance.calculate(&data.row(i), &data.row(j));
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let knn_set: HashSet<usize> =
                distances.iter().take(self.k).map(|&(idx, _)| idx).collect();
            knn_sets.push(knn_set);
        }

        // Add edges only for mutual k-nearest neighbors
        for i in 0..n_samples {
            for &j in &knn_sets[i] {
                if knn_sets[j].contains(&i) {
                    let dist = self.distance.calculate(&data.row(i), &data.row(j));
                    graph.add_edge(i, j, dist);
                }
            }
        }

        Ok(graph)
    }
}

/// Epsilon Graph Builder (Radius Neighbors Graph)
pub struct EpsilonGraph {
    epsilon: Float,
    distance: Distance,
    directed: bool,
}

impl EpsilonGraph {
    /// Create a new epsilon graph builder
    pub fn new(epsilon: Float) -> Self {
        Self {
            epsilon,
            distance: Distance::Euclidean,
            directed: false,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set whether the graph should be directed
    pub fn with_directed(mut self, directed: bool) -> Self {
        self.directed = directed;
        self
    }

    /// Build the epsilon graph
    pub fn build(&self, data: &ArrayView2<Float>) -> NeighborsResult<NeighborhoodGraph> {
        let n_samples = data.nrows();

        if n_samples == 0 {
            return Err(NeighborsError::EmptyInput);
        }

        if self.epsilon <= 0.0 {
            return Err(NeighborsError::InvalidRadius(self.epsilon));
        }

        let mut graph = NeighborhoodGraph::new(n_samples);

        // For each pair of samples, add edge if distance <= epsilon
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist = self.distance.calculate(&data.row(i), &data.row(j));

                if dist <= self.epsilon {
                    graph.add_edge(i, j, dist);

                    if !self.directed {
                        graph.add_edge(j, i, dist);
                    }
                }
            }
        }

        Ok(graph)
    }
}

/// Relative Neighborhood Graph Builder
pub struct RelativeNeighborhoodGraph {
    distance: Distance,
}

impl Default for RelativeNeighborhoodGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl RelativeNeighborhoodGraph {
    /// Create a new relative neighborhood graph builder
    pub fn new() -> Self {
        Self {
            distance: Distance::Euclidean,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Build the relative neighborhood graph
    pub fn build(&self, data: &ArrayView2<Float>) -> NeighborsResult<NeighborhoodGraph> {
        let n_samples = data.nrows();

        if n_samples == 0 {
            return Err(NeighborsError::EmptyInput);
        }

        let mut graph = NeighborhoodGraph::new(n_samples);

        // For each pair of points, check if they are relative neighbors
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist_ij = self.distance.calculate(&data.row(i), &data.row(j));
                let mut is_relative_neighbor = true;

                // Check if there's any point k such that max(d(i,k), d(j,k)) < d(i,j)
                for k in 0..n_samples {
                    if k != i && k != j {
                        let dist_ik = self.distance.calculate(&data.row(i), &data.row(k));
                        let dist_jk = self.distance.calculate(&data.row(j), &data.row(k));

                        if dist_ik.max(dist_jk) < dist_ij {
                            is_relative_neighbor = false;
                            break;
                        }
                    }
                }

                if is_relative_neighbor {
                    graph.add_edge(i, j, dist_ij);
                    graph.add_edge(j, i, dist_ij);
                }
            }
        }

        Ok(graph)
    }
}

/// Gabriel Graph Builder
pub struct GabrielGraph {
    distance: Distance,
}

impl Default for GabrielGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl GabrielGraph {
    /// Create a new Gabriel graph builder
    pub fn new() -> Self {
        Self {
            distance: Distance::Euclidean,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Build the Gabriel graph
    pub fn build(&self, data: &ArrayView2<Float>) -> NeighborsResult<NeighborhoodGraph> {
        let n_samples = data.nrows();

        if n_samples == 0 {
            return Err(NeighborsError::EmptyInput);
        }

        let mut graph = NeighborhoodGraph::new(n_samples);

        // For each pair of points, check if they form a Gabriel edge
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist_ij = self.distance.calculate(&data.row(i), &data.row(j));
                let mut is_gabriel_edge = true;

                // Check if there's any point k inside the circle with diameter ij
                for k in 0..n_samples {
                    if k != i && k != j {
                        let dist_ik = self.distance.calculate(&data.row(i), &data.row(k));
                        let dist_jk = self.distance.calculate(&data.row(j), &data.row(k));

                        // Check if k is inside the circle with diameter ij
                        // This is true if d(i,k)^2 + d(j,k)^2 < d(i,j)^2
                        if dist_ik * dist_ik + dist_jk * dist_jk < dist_ij * dist_ij {
                            is_gabriel_edge = false;
                            break;
                        }
                    }
                }

                if is_gabriel_edge {
                    graph.add_edge(i, j, dist_ij);
                    graph.add_edge(j, i, dist_ij);
                }
            }
        }

        Ok(graph)
    }
}

/// Graph-based neighbor search utilities
pub struct GraphNeighborSearch {
    graph: NeighborhoodGraph,
}

impl GraphNeighborSearch {
    /// Create a new graph-based neighbor search
    pub fn new(graph: NeighborhoodGraph) -> Self {
        Self { graph }
    }

    /// Find all neighbors of a node
    pub fn find_neighbors(&self, node: usize) -> Vec<usize> {
        self.graph.get_neighbors(node)
    }

    /// Find neighbors within a given number of hops
    pub fn find_neighbors_within_hops(&self, node: usize, hops: usize) -> Vec<usize> {
        if hops == 0 {
            return vec![node];
        }

        let mut visited = HashSet::new();
        let mut current_level = vec![node];
        visited.insert(node);

        for _ in 0..hops {
            let mut next_level = Vec::new();

            for &current_node in &current_level {
                for neighbor in self.graph.get_neighbors(current_node) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        next_level.push(neighbor);
                    }
                }
            }

            current_level = next_level;
        }

        visited.into_iter().collect()
    }

    /// Find shortest path between two nodes using BFS
    pub fn shortest_path(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        if start >= self.graph.nodes || end >= self.graph.nodes {
            return None;
        }

        if start == end {
            return Some(vec![start]);
        }

        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut parent = HashMap::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if current == end {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = end;

                while let Some(&prev) = parent.get(&node) {
                    path.push(node);
                    node = prev;
                }
                path.push(start);
                path.reverse();

                return Some(path);
            }

            for neighbor in self.graph.get_neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Check if the graph is connected
    pub fn is_connected(&self) -> bool {
        if self.graph.nodes == 0 {
            return true;
        }

        let reachable = self.find_neighbors_within_hops(0, self.graph.nodes);
        reachable.len() == self.graph.nodes
    }

    /// Get connected components
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for node in 0..self.graph.nodes {
            if !visited.contains(&node) {
                let component = self.find_connected_component(node, &mut visited);
                components.push(component);
            }
        }

        components
    }

    /// Find connected component containing a specific node
    fn find_connected_component(&self, start: usize, visited: &mut HashSet<usize>) -> Vec<usize> {
        let mut component = Vec::new();
        let mut stack = vec![start];

        while let Some(node) = stack.pop() {
            if !visited.contains(&node) {
                visited.insert(node);
                component.push(node);

                for neighbor in self.graph.get_neighbors(node) {
                    if !visited.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }

        component
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn create_test_data() -> Array2<Float> {
        Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn test_knn_graph() {
        let data = create_test_data();
        let graph_builder = KNearestNeighborGraph::new(2);
        let graph = graph_builder.build(&data.view()).unwrap();

        assert_eq!(graph.nodes, 5);
        assert!(graph.edges.len() > 0);

        // Each node should have at most 2 outgoing edges
        for i in 0..graph.nodes {
            assert!(graph.degree(i) <= 2);
        }
    }

    #[test]
    fn test_mutual_knn_graph() {
        let data = create_test_data();
        let graph_builder = MutualKNearestNeighbors::new(2);
        let graph = graph_builder.build(&data.view()).unwrap();

        assert_eq!(graph.nodes, 5);

        // Check that edges are mutual
        for edge in &graph.edges {
            assert!(graph.has_edge(edge.target, edge.source));
        }
    }

    #[test]
    fn test_epsilon_graph() {
        let data = create_test_data();
        let graph_builder = EpsilonGraph::new(1.5);
        let graph = graph_builder.build(&data.view()).unwrap();

        assert_eq!(graph.nodes, 5);

        // Check that all edges have distance <= epsilon
        for edge in &graph.edges {
            assert!(edge.weight <= 1.5);
        }
    }

    #[test]
    fn test_relative_neighborhood_graph() {
        let data = create_test_data();
        let graph_builder = RelativeNeighborhoodGraph::new();
        let graph = graph_builder.build(&data.view()).unwrap();

        assert_eq!(graph.nodes, 5);

        // RNG should be a subgraph of Delaunay triangulation
        assert!(graph.edges.len() <= data.nrows() * (data.nrows() - 1) / 2);
    }

    #[test]
    fn test_gabriel_graph() {
        let data = create_test_data();
        let graph_builder = GabrielGraph::new();
        let graph = graph_builder.build(&data.view()).unwrap();

        assert_eq!(graph.nodes, 5);

        // Gabriel graph may include both directions for each edge, so allow for directed graph
        let max_directed_edges = data.nrows() * (data.nrows() - 1);
        assert!(graph.edges.len() <= max_directed_edges);
    }

    #[test]
    fn test_graph_neighbor_search() {
        let data = create_test_data();
        let graph_builder = KNearestNeighborGraph::new(2).with_directed(false);
        let graph = graph_builder.build(&data.view()).unwrap();

        let search = GraphNeighborSearch::new(graph);

        // Test finding neighbors
        let neighbors = search.find_neighbors(0);
        assert!(neighbors.len() > 0);

        // Test shortest path
        let path = search.shortest_path(0, 4);
        assert!(path.is_some());

        // Test connected components
        let components = search.connected_components();
        assert!(components.len() > 0);
    }

    #[test]
    fn test_graph_statistics() {
        let data = create_test_data();
        let graph_builder = KNearestNeighborGraph::new(2);
        let graph = graph_builder.build(&data.view()).unwrap();

        let stats = graph.statistics();
        assert_eq!(stats.nodes, 5);
        assert!(stats.edges > 0);
        assert!(stats.max_degree >= stats.min_degree);
        assert!(stats.avg_degree >= 0.0);
    }
}
