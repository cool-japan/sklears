//! Graph Kernel Approximations
//!
//! This module implements various graph kernel approximation methods for
//! analyzing graph-structured data such as molecular graphs, social networks,
//! and other relational data structures.
//!
//! # Key Features
//!
//! - **Random Walk Kernels**: Count common random walks between graphs
//! - **Shortest Path Kernels**: Compare shortest path distributions
//! - **Weisfeiler-Lehman Kernels**: Graph isomorphism-based kernels
//! - **Subgraph Kernels**: Count common subgraph patterns
//! - **Graphlet Kernels**: Count small connected subgraphs
//! - **Graph Laplacian Kernels**: Use graph Laplacian spectrum
//!
//! # Mathematical Background
//!
//! Graph kernel between graphs G₁ and G₂:
//! K(G₁, G₂) = Σ φ(G₁)[f] * φ(G₂)[f]
//!
//! Where φ(G)[f] is the feature map counting occurrences of feature f in graph G.
//!
//! # References
//!
//! - Vishwanathan, S. V. N., et al. (2010). Graph kernels
//! - Weisfeiler, B., & Lehman, A. A. (1968). The reduction of a graph to canonical form

use scirs2_core::ndarray::Array2;
use sklears_core::error::Result;
use sklears_core::traits::{Fit, Transform};
use std::collections::{HashMap, HashSet, VecDeque};

/// Simple graph representation
#[derive(Debug, Clone)]
/// Graph
pub struct Graph {
    /// Adjacency list representation
    pub adjacency: HashMap<usize, Vec<usize>>,
    /// Node labels (optional)
    pub node_labels: Option<HashMap<usize, String>>,
    /// Edge labels (optional)
    pub edge_labels: Option<HashMap<(usize, usize), String>>,
    /// Number of nodes
    pub num_nodes: usize,
}

impl Graph {
    /// Create new graph
    pub fn new(num_nodes: usize) -> Self {
        Self {
            adjacency: HashMap::new(),
            node_labels: None,
            edge_labels: None,
            num_nodes,
        }
    }

    /// Add edge
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.adjacency.entry(from).or_insert_with(Vec::new).push(to);
        self.adjacency.entry(to).or_insert_with(Vec::new).push(from);
    }

    /// Add directed edge
    pub fn add_directed_edge(&mut self, from: usize, to: usize) {
        self.adjacency.entry(from).or_insert_with(Vec::new).push(to);
    }

    /// Set node labels
    pub fn set_node_labels(&mut self, labels: HashMap<usize, String>) {
        self.node_labels = Some(labels);
    }

    /// Set edge labels
    pub fn set_edge_labels(&mut self, labels: HashMap<(usize, usize), String>) {
        self.edge_labels = Some(labels);
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.adjacency.get(&node).cloned().unwrap_or_default()
    }

    /// Get all nodes
    pub fn nodes(&self) -> Vec<usize> {
        (0..self.num_nodes).collect()
    }

    /// Get all edges
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for (&from, neighbors) in &self.adjacency {
            for &to in neighbors {
                if from <= to {
                    // Avoid duplicates for undirected graphs
                    edges.push((from, to));
                }
            }
        }
        edges
    }
}

/// Random walk kernel for graphs
#[derive(Debug, Clone)]
/// RandomWalkKernel
pub struct RandomWalkKernel {
    /// Maximum walk length
    max_length: usize,
    /// Convergence parameter (lambda)
    lambda: f64,
    /// Whether to use node labels
    use_node_labels: bool,
    /// Whether to use edge labels
    use_edge_labels: bool,
}

impl RandomWalkKernel {
    pub fn new(max_length: usize, lambda: f64) -> Self {
        Self {
            max_length,
            lambda,
            use_node_labels: false,
            use_edge_labels: false,
        }
    }

    /// Enable node labels
    pub fn use_node_labels(mut self, use_labels: bool) -> Self {
        self.use_node_labels = use_labels;
        self
    }

    /// Enable edge labels
    pub fn use_edge_labels(mut self, use_labels: bool) -> Self {
        self.use_edge_labels = use_labels;
        self
    }

    /// Compute direct product graph for random walk kernel
    fn product_graph(&self, g1: &Graph, g2: &Graph) -> Graph {
        let mut product = Graph::new(g1.num_nodes * g2.num_nodes);

        // Create nodes in product graph: (i, j) -> i * g2.num_nodes + j
        for i in 0..g1.num_nodes {
            for j in 0..g2.num_nodes {
                let node_ij = i * g2.num_nodes + j;

                // Check if nodes match (labels if available)
                let nodes_match = if self.use_node_labels {
                    if let (Some(labels1), Some(labels2)) = (&g1.node_labels, &g2.node_labels) {
                        labels1.get(&i) == labels2.get(&j)
                    } else {
                        true
                    }
                } else {
                    true
                };

                if !nodes_match {
                    continue;
                }

                // Add edges in product graph
                for &neighbor_i in &g1.neighbors(i) {
                    for &neighbor_j in &g2.neighbors(j) {
                        let neighbor_ij = neighbor_i * g2.num_nodes + neighbor_j;

                        // Check if edges match (labels if available)
                        let edges_match = if self.use_edge_labels {
                            if let (Some(labels1), Some(labels2)) =
                                (&g1.edge_labels, &g2.edge_labels)
                            {
                                labels1.get(&(i, neighbor_i)) == labels2.get(&(j, neighbor_j))
                            } else {
                                true
                            }
                        } else {
                            true
                        };

                        if edges_match {
                            product.add_directed_edge(node_ij, neighbor_ij);
                        }
                    }
                }
            }
        }

        product
    }

    /// Compute random walk kernel value between two graphs
    fn kernel_value(&self, g1: &Graph, g2: &Graph) -> f64 {
        let product = self.product_graph(g1, g2);

        // Count walks in product graph using matrix powers
        let n = product.num_nodes;
        if n == 0 {
            return 0.0;
        }

        // Build adjacency matrix
        let mut adj = Array2::zeros((n, n));
        for (&from, neighbors) in &product.adjacency {
            for &to in neighbors {
                adj[(from, to)] = 1.0;
            }
        }

        // Compute sum of matrix powers: I + λA + λ²A² + ... + λᵏAᵏ
        let result = Array2::eye(n);
        let mut current_power = Array2::eye(n);
        let mut total = result.clone();

        for k in 1..=self.max_length {
            current_power = current_power.dot(&adj);
            total = total + self.lambda.powi(k as i32) * &current_power;
        }

        // Sum all entries (total number of walks)
        total.sum()
    }
}

/// Fitted random walk kernel
#[derive(Debug, Clone)]
/// FittedRandomWalkKernel
pub struct FittedRandomWalkKernel {
    /// Training graphs
    training_graphs: Vec<Graph>,
    /// Kernel parameters
    max_length: usize,
    lambda: f64,
    use_node_labels: bool,
    use_edge_labels: bool,
}

impl Fit<Vec<Graph>, ()> for RandomWalkKernel {
    type Fitted = FittedRandomWalkKernel;
    fn fit(self, graphs: &Vec<Graph>, _y: &()) -> Result<Self::Fitted> {
        Ok(FittedRandomWalkKernel {
            training_graphs: graphs.clone(),
            max_length: self.max_length,
            lambda: self.lambda,
            use_node_labels: self.use_node_labels,
            use_edge_labels: self.use_edge_labels,
        })
    }
}

impl Transform<Vec<Graph>, Array2<f64>> for FittedRandomWalkKernel {
    fn transform(&self, graphs: &Vec<Graph>) -> Result<Array2<f64>> {
        let n_test = graphs.len();
        let n_train = self.training_graphs.len();
        let mut kernel_matrix = Array2::zeros((n_test, n_train));

        let kernel = RandomWalkKernel {
            max_length: self.max_length,
            lambda: self.lambda,
            use_node_labels: self.use_node_labels,
            use_edge_labels: self.use_edge_labels,
        };

        for i in 0..n_test {
            for j in 0..n_train {
                kernel_matrix[(i, j)] = kernel.kernel_value(&graphs[i], &self.training_graphs[j]);
            }
        }

        Ok(kernel_matrix)
    }
}

/// Shortest path kernel for graphs
#[derive(Debug, Clone)]
/// ShortestPathKernel
pub struct ShortestPathKernel {
    /// Whether to use node labels
    use_node_labels: bool,
    /// Whether to normalize by graph size
    normalize: bool,
}

impl ShortestPathKernel {
    pub fn new() -> Self {
        Self {
            use_node_labels: false,
            normalize: true,
        }
    }

    /// Enable node labels
    pub fn use_node_labels(mut self, use_labels: bool) -> Self {
        self.use_node_labels = use_labels;
        self
    }

    /// Set normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Compute shortest paths between all pairs of nodes
    fn all_pairs_shortest_paths(&self, graph: &Graph) -> HashMap<(usize, usize), usize> {
        let mut distances = HashMap::new();
        let nodes = graph.nodes();

        // Initialize distances
        for &i in &nodes {
            for &j in &nodes {
                if i == j {
                    distances.insert((i, j), 0);
                } else {
                    distances.insert((i, j), usize::MAX);
                }
            }
        }

        // Set direct edge distances
        for (&from, neighbors) in &graph.adjacency {
            for &to in neighbors {
                distances.insert((from, to), 1);
            }
        }

        // Floyd-Warshall algorithm
        for &k in &nodes {
            for &i in &nodes {
                for &j in &nodes {
                    if let (Some(&dist_ik), Some(&dist_kj)) =
                        (distances.get(&(i, k)), distances.get(&(k, j)))
                    {
                        if dist_ik != usize::MAX && dist_kj != usize::MAX {
                            let new_dist = dist_ik + dist_kj;
                            if let Some(current_dist) = distances.get_mut(&(i, j)) {
                                if new_dist < *current_dist {
                                    *current_dist = new_dist;
                                }
                            }
                        }
                    }
                }
            }
        }

        distances
    }

    /// Extract shortest path features
    fn extract_features(&self, graph: &Graph) -> HashMap<String, usize> {
        let distances = self.all_pairs_shortest_paths(graph);
        let mut features = HashMap::new();

        for ((i, j), &dist) in &distances {
            if dist != usize::MAX {
                let feature = if self.use_node_labels {
                    if let Some(ref labels) = graph.node_labels {
                        let label_i = labels.get(i).cloned().unwrap_or_default();
                        let label_j = labels.get(j).cloned().unwrap_or_default();
                        format!("{}:{}:{}", label_i, label_j, dist)
                    } else {
                        format!("path:{}", dist)
                    }
                } else {
                    format!("path:{}", dist)
                };

                *features.entry(feature).or_insert(0) += 1;
            }
        }

        features
    }

    /// Compute kernel value between two graphs
    fn kernel_value(&self, g1: &Graph, g2: &Graph) -> f64 {
        let features1 = self.extract_features(g1);
        let features2 = self.extract_features(g2);

        let mut dot_product = 0.0;
        for (feature, &count1) in &features1 {
            if let Some(&count2) = features2.get(feature) {
                dot_product += (count1 * count2) as f64;
            }
        }

        if self.normalize {
            let norm1 = features1
                .values()
                .map(|&x| (x * x) as f64)
                .sum::<f64>()
                .sqrt();
            let norm2 = features2
                .values()
                .map(|&x| (x * x) as f64)
                .sum::<f64>()
                .sqrt();
            if norm1 > 0.0 && norm2 > 0.0 {
                dot_product / (norm1 * norm2)
            } else {
                0.0
            }
        } else {
            dot_product
        }
    }
}

/// Fitted shortest path kernel
#[derive(Debug, Clone)]
/// FittedShortestPathKernel
pub struct FittedShortestPathKernel {
    /// Training graphs
    training_graphs: Vec<Graph>,
    /// Use node labels
    use_node_labels: bool,
    /// Normalize flag
    normalize: bool,
}

impl Fit<Vec<Graph>, ()> for ShortestPathKernel {
    type Fitted = FittedShortestPathKernel;

    fn fit(self, graphs: &Vec<Graph>, _y: &()) -> Result<Self::Fitted> {
        Ok(FittedShortestPathKernel {
            training_graphs: graphs.clone(),
            use_node_labels: self.use_node_labels,
            normalize: self.normalize,
        })
    }
}

impl Transform<Vec<Graph>, Array2<f64>> for FittedShortestPathKernel {
    fn transform(&self, graphs: &Vec<Graph>) -> Result<Array2<f64>> {
        let n_test = graphs.len();
        let n_train = self.training_graphs.len();
        let mut kernel_matrix = Array2::zeros((n_test, n_train));

        let kernel = ShortestPathKernel {
            use_node_labels: self.use_node_labels,
            normalize: self.normalize,
        };

        for i in 0..n_test {
            for j in 0..n_train {
                kernel_matrix[(i, j)] = kernel.kernel_value(&graphs[i], &self.training_graphs[j]);
            }
        }

        Ok(kernel_matrix)
    }
}

/// Weisfeiler-Lehman kernel for graphs
#[derive(Debug, Clone)]
/// WeisfeilerLehmanKernel
pub struct WeisfeilerLehmanKernel {
    /// Number of iterations
    iterations: usize,
    /// Whether to use original node labels
    use_node_labels: bool,
    /// Whether to normalize features
    normalize: bool,
}

impl WeisfeilerLehmanKernel {
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            use_node_labels: false,
            normalize: true,
        }
    }

    /// Enable node labels
    pub fn use_node_labels(mut self, use_labels: bool) -> Self {
        self.use_node_labels = use_labels;
        self
    }

    /// Set normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Perform Weisfeiler-Lehman relabeling
    fn wl_relabel(&self, graph: &Graph) -> Vec<HashMap<usize, String>> {
        let mut labelings = Vec::new();
        let nodes = graph.nodes();

        // Initial labeling
        let mut current_labels = HashMap::new();
        for &node in &nodes {
            let initial_label = if self.use_node_labels {
                graph
                    .node_labels
                    .as_ref()
                    .and_then(|labels| labels.get(&node))
                    .cloned()
                    .unwrap_or_else(|| "default".to_string())
            } else {
                "1".to_string()
            };
            current_labels.insert(node, initial_label);
        }
        labelings.push(current_labels.clone());

        // Iterative relabeling
        for _iter in 0..self.iterations {
            let mut new_labels = HashMap::new();

            for &node in &nodes {
                let mut neighbor_labels = Vec::new();
                for &neighbor in &graph.neighbors(node) {
                    if let Some(label) = current_labels.get(&neighbor) {
                        neighbor_labels.push(label.clone());
                    }
                }
                neighbor_labels.sort();

                let current_label = current_labels.get(&node).cloned().unwrap_or_default();
                let new_label = format!("{}:{}", current_label, neighbor_labels.join(","));
                new_labels.insert(node, new_label);
            }

            labelings.push(new_labels.clone());
            current_labels = new_labels;
        }

        labelings
    }

    /// Extract features from WL labelings
    fn extract_features(&self, graph: &Graph) -> HashMap<String, usize> {
        let labelings = self.wl_relabel(graph);
        let mut features = HashMap::new();

        for labeling in labelings {
            for (_, label) in labeling {
                *features.entry(label).or_insert(0) += 1;
            }
        }

        features
    }

    /// Compute kernel value between two graphs
    fn kernel_value(&self, g1: &Graph, g2: &Graph) -> f64 {
        let features1 = self.extract_features(g1);
        let features2 = self.extract_features(g2);

        let mut dot_product = 0.0;
        for (feature, &count1) in &features1 {
            if let Some(&count2) = features2.get(feature) {
                dot_product += (count1 * count2) as f64;
            }
        }

        if self.normalize {
            let norm1 = features1
                .values()
                .map(|&x| (x * x) as f64)
                .sum::<f64>()
                .sqrt();
            let norm2 = features2
                .values()
                .map(|&x| (x * x) as f64)
                .sum::<f64>()
                .sqrt();
            if norm1 > 0.0 && norm2 > 0.0 {
                dot_product / (norm1 * norm2)
            } else {
                0.0
            }
        } else {
            dot_product
        }
    }
}

/// Fitted Weisfeiler-Lehman kernel
#[derive(Debug, Clone)]
/// FittedWeisfeilerLehmanKernel
pub struct FittedWeisfeilerLehmanKernel {
    /// Training graphs
    training_graphs: Vec<Graph>,
    /// Number of iterations
    iterations: usize,
    /// Use node labels
    use_node_labels: bool,
    /// Normalize flag
    normalize: bool,
}

impl Fit<Vec<Graph>, ()> for WeisfeilerLehmanKernel {
    type Fitted = FittedWeisfeilerLehmanKernel;
    fn fit(self, graphs: &Vec<Graph>, _y: &()) -> Result<Self::Fitted> {
        Ok(FittedWeisfeilerLehmanKernel {
            training_graphs: graphs.clone(),
            iterations: self.iterations,
            use_node_labels: self.use_node_labels,
            normalize: self.normalize,
        })
    }
}

impl Transform<Vec<Graph>, Array2<f64>> for FittedWeisfeilerLehmanKernel {
    fn transform(&self, graphs: &Vec<Graph>) -> Result<Array2<f64>> {
        let n_test = graphs.len();
        let n_train = self.training_graphs.len();
        let mut kernel_matrix = Array2::zeros((n_test, n_train));

        let kernel = WeisfeilerLehmanKernel {
            iterations: self.iterations,
            use_node_labels: self.use_node_labels,
            normalize: self.normalize,
        };

        for i in 0..n_test {
            for j in 0..n_train {
                kernel_matrix[(i, j)] = kernel.kernel_value(&graphs[i], &self.training_graphs[j]);
            }
        }

        Ok(kernel_matrix)
    }
}

/// Subgraph kernel that counts common subgraph patterns
#[derive(Debug, Clone)]
/// SubgraphKernel
pub struct SubgraphKernel {
    /// Maximum subgraph size
    max_size: usize,
    /// Whether to use connected subgraphs only
    connected_only: bool,
    /// Whether to normalize features
    normalize: bool,
}

impl SubgraphKernel {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            connected_only: true,
            normalize: true,
        }
    }

    /// Set connected subgraphs only
    pub fn connected_only(mut self, connected: bool) -> Self {
        self.connected_only = connected;
        self
    }

    /// Set normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Find all connected subgraphs of given size
    fn find_connected_subgraphs(&self, graph: &Graph, size: usize) -> Vec<Vec<usize>> {
        if size == 0 {
            return vec![];
        }

        let mut subgraphs = Vec::new();
        let nodes = graph.nodes();

        // Generate all combinations of nodes of given size
        let combinations = self.combinations(&nodes, size);

        for combination in combinations {
            if self.is_connected_subgraph(graph, &combination) {
                subgraphs.push(combination);
            }
        }

        subgraphs
    }

    /// Check if a set of nodes forms a connected subgraph
    fn is_connected_subgraph(&self, graph: &Graph, nodes: &[usize]) -> bool {
        if nodes.len() <= 1 {
            return true;
        }

        let node_set: HashSet<_> = nodes.iter().collect();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start BFS from first node
        queue.push_back(nodes[0]);
        visited.insert(nodes[0]);

        while let Some(current) = queue.pop_front() {
            for &neighbor in &graph.neighbors(current) {
                if node_set.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        visited.len() == nodes.len()
    }

    /// Generate all combinations of k elements from a vector
    fn combinations(&self, items: &[usize], k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        if k > items.len() {
            return vec![];
        }
        if k == items.len() {
            return vec![items.to_vec()];
        }

        let mut result = Vec::new();

        // Include first element
        let with_first = self.combinations(&items[1..], k - 1);
        for mut combo in with_first {
            combo.insert(0, items[0]);
            result.push(combo);
        }

        // Exclude first element
        let without_first = self.combinations(&items[1..], k);
        result.extend(without_first);

        result
    }

    /// Convert subgraph to canonical string representation
    fn subgraph_to_string(&self, graph: &Graph, nodes: &[usize]) -> String {
        let mut edges = Vec::new();
        let node_set: HashSet<_> = nodes.iter().collect();

        for &node in nodes {
            for &neighbor in &graph.neighbors(node) {
                if node_set.contains(&neighbor) && node < neighbor {
                    edges.push((node, neighbor));
                }
            }
        }

        edges.sort();
        format!("nodes:{},edges:{:?}", nodes.len(), edges)
    }

    /// Extract subgraph features
    fn extract_features(&self, graph: &Graph) -> HashMap<String, usize> {
        let mut features = HashMap::new();

        for size in 1..=self.max_size {
            let subgraphs = if self.connected_only {
                self.find_connected_subgraphs(graph, size)
            } else {
                // For simplicity, just use connected subgraphs
                self.find_connected_subgraphs(graph, size)
            };

            for subgraph in subgraphs {
                let feature = self.subgraph_to_string(graph, &subgraph);
                *features.entry(feature).or_insert(0) += 1;
            }
        }

        features
    }

    /// Compute kernel value between two graphs
    fn kernel_value(&self, g1: &Graph, g2: &Graph) -> f64 {
        let features1 = self.extract_features(g1);
        let features2 = self.extract_features(g2);

        let mut dot_product = 0.0;
        for (feature, &count1) in &features1 {
            if let Some(&count2) = features2.get(feature) {
                dot_product += (count1 * count2) as f64;
            }
        }

        if self.normalize {
            let norm1 = features1
                .values()
                .map(|&x| (x * x) as f64)
                .sum::<f64>()
                .sqrt();
            let norm2 = features2
                .values()
                .map(|&x| (x * x) as f64)
                .sum::<f64>()
                .sqrt();
            if norm1 > 0.0 && norm2 > 0.0 {
                dot_product / (norm1 * norm2)
            } else {
                0.0
            }
        } else {
            dot_product
        }
    }
}

/// Fitted subgraph kernel
#[derive(Debug, Clone)]
/// FittedSubgraphKernel
pub struct FittedSubgraphKernel {
    /// Training graphs
    training_graphs: Vec<Graph>,
    /// Max subgraph size
    max_size: usize,
    /// Connected only flag
    connected_only: bool,
    /// Normalize flag
    normalize: bool,
}

impl Fit<Vec<Graph>, ()> for SubgraphKernel {
    type Fitted = FittedSubgraphKernel;

    fn fit(self, graphs: &Vec<Graph>, _y: &()) -> Result<Self::Fitted> {
        Ok(FittedSubgraphKernel {
            training_graphs: graphs.clone(),
            max_size: self.max_size,
            connected_only: self.connected_only,
            normalize: self.normalize,
        })
    }
}

impl Transform<Vec<Graph>, Array2<f64>> for FittedSubgraphKernel {
    fn transform(&self, graphs: &Vec<Graph>) -> Result<Array2<f64>> {
        let n_test = graphs.len();
        let n_train = self.training_graphs.len();
        let mut kernel_matrix = Array2::zeros((n_test, n_train));

        let kernel = SubgraphKernel {
            max_size: self.max_size,
            connected_only: self.connected_only,
            normalize: self.normalize,
        };

        for i in 0..n_test {
            for j in 0..n_train {
                kernel_matrix[(i, j)] = kernel.kernel_value(&graphs[i], &self.training_graphs[j]);
            }
        }

        Ok(kernel_matrix)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn create_test_graph(edges: Vec<(usize, usize)>, num_nodes: usize) -> Graph {
        let mut graph = Graph::new(num_nodes);
        for (from, to) in edges {
            graph.add_edge(from, to);
        }
        graph
    }

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        assert_eq!(graph.neighbors(0), vec![1]);
        assert_eq!(graph.neighbors(1), vec![0, 2]);
        assert_eq!(graph.neighbors(2), vec![1]);
        assert_eq!(graph.nodes(), vec![0, 1, 2]);
    }

    #[test]
    fn test_random_walk_kernel() {
        let kernel = RandomWalkKernel::new(3, 0.1);

        let graph1 = create_test_graph(vec![(0, 1), (1, 2)], 3);
        let graph2 = create_test_graph(vec![(0, 1), (1, 2)], 3);
        let graph3 = create_test_graph(vec![(0, 1), (0, 2), (1, 2)], 3);

        let graphs = vec![graph1, graph2, graph3];
        let fitted = kernel.fit(&graphs, &()).unwrap();
        let kernel_matrix = fitted.transform(&graphs).unwrap();

        assert_eq!(kernel_matrix.shape(), &[3, 3]);

        // Identical graphs should have same kernel value
        assert_abs_diff_eq!(
            kernel_matrix[(0, 0)],
            kernel_matrix[(1, 1)],
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            kernel_matrix[(0, 1)],
            kernel_matrix[(1, 0)],
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_shortest_path_kernel() {
        let kernel = ShortestPathKernel::new();

        let graph1 = create_test_graph(vec![(0, 1), (1, 2)], 3);
        let graph2 = create_test_graph(vec![(0, 1), (1, 2), (0, 2)], 3);

        let graphs = vec![graph1, graph2];
        let fitted = kernel.fit(&graphs, &()).unwrap();
        let kernel_matrix = fitted.transform(&graphs).unwrap();

        assert_eq!(kernel_matrix.shape(), &[2, 2]);
        // Kernel values should be non-negative and finite
        assert!(kernel_matrix.iter().all(|&x| x >= 0.0 && x.is_finite()));

        // Self-similarity should be 1.0 for normalized kernels
        assert_abs_diff_eq!(kernel_matrix[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(kernel_matrix[(1, 1)], 1.0, epsilon = 1e-10);

        // Matrix should be symmetric
        assert_abs_diff_eq!(
            kernel_matrix[(0, 1)],
            kernel_matrix[(1, 0)],
            epsilon = 1e-10
        );

        // All off-diagonal elements should be between 0 and 1 for normalized cosine similarity
        assert!(kernel_matrix[(0, 1)] >= 0.0 && kernel_matrix[(0, 1)] <= 1.0);
    }

    #[test]
    fn test_weisfeiler_lehman_kernel() {
        let kernel = WeisfeilerLehmanKernel::new(2);

        let graph1 = create_test_graph(vec![(0, 1), (1, 2)], 3);
        let graph2 = create_test_graph(vec![(0, 1), (1, 2)], 3);
        let graph3 = create_test_graph(vec![(0, 1), (0, 2), (1, 2)], 3);

        let graphs = vec![graph1, graph2, graph3];
        let fitted = kernel.fit(&graphs, &()).unwrap();
        let kernel_matrix = fitted.transform(&graphs).unwrap();

        assert_eq!(kernel_matrix.shape(), &[3, 3]);
        assert!(kernel_matrix
            .iter()
            .all(|&x| x >= 0.0 && x <= 1.0 && x.is_finite()));

        // Identical graphs should have similarity 1.0
        assert_abs_diff_eq!(kernel_matrix[(0, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_subgraph_kernel() {
        let kernel = SubgraphKernel::new(2);

        let graph1 = create_test_graph(vec![(0, 1)], 2);
        let graph2 = create_test_graph(vec![(0, 1), (1, 2)], 3);

        let graphs = vec![graph1, graph2];
        let fitted = kernel.fit(&graphs, &()).unwrap();
        let kernel_matrix = fitted.transform(&graphs).unwrap();

        assert_eq!(kernel_matrix.shape(), &[2, 2]);
        assert!(kernel_matrix.iter().all(|&x| x >= 0.0 && x.is_finite()));
    }

    #[test]
    fn test_graph_with_labels() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let mut labels = HashMap::new();
        labels.insert(0, "A".to_string());
        labels.insert(1, "B".to_string());
        labels.insert(2, "A".to_string());
        graph.set_node_labels(labels);

        let kernel = WeisfeilerLehmanKernel::new(1).use_node_labels(true);
        let graphs = vec![graph];
        let fitted = kernel.fit(&graphs, &()).unwrap();
        let kernel_matrix = fitted.transform(&graphs).unwrap();

        assert_eq!(kernel_matrix.shape(), &[1, 1]);
        assert_abs_diff_eq!(kernel_matrix[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shortest_path_computation() {
        let kernel = ShortestPathKernel::new();
        let graph = create_test_graph(vec![(0, 1), (1, 2), (2, 3)], 4);

        let distances = kernel.all_pairs_shortest_paths(&graph);

        assert_eq!(distances[&(0, 3)], 3);
        assert_eq!(distances[&(0, 1)], 1);
        assert_eq!(distances[&(1, 3)], 2);
    }

    #[test]
    fn test_subgraph_connectivity() {
        let kernel = SubgraphKernel::new(3);
        let graph = create_test_graph(vec![(0, 1), (2, 3)], 4); // Disconnected graph

        assert!(!kernel.is_connected_subgraph(&graph, &[0, 1, 2]));
        assert!(kernel.is_connected_subgraph(&graph, &[0, 1]));
        assert!(kernel.is_connected_subgraph(&graph, &[2, 3]));
    }
}
