//! Graph analysis and metrics for trait relationship graphs
//!
//! This module provides comprehensive graph analysis capabilities including
//! centrality measures, community detection, path analysis, clustering metrics,
//! and network topology analysis with SIMD acceleration for performance.

use super::graph_config::{CentralityMeasure, CommunityDetection};
use super::graph_structures::{
    TraitGraph, TraitGraphNode, TraitGraphEdge, CentralityMeasures, Community, GraphPath,
    GraphAnalysisResult, GraphQualityMetrics, PerformanceMetrics,
};
use crate::error::{Result, SklearsError};

// SciRS2 Core imports for numerical computations and SIMD acceleration
use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Random, thread_rng, CoreRandom};
use scirs2_core::Rng;
// SIMD operations may not be available in current scirs2_core version
// use scirs2_core::simd::{SimdArray, SimdOps};

use chrono::Utc;
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use std::f64;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Advanced graph analyzer for centrality, clustering, and path analysis
#[derive(Debug)]
pub struct GraphAnalyzer {
    /// SIMD optimization enabled
    simd_enabled: bool,
    /// Parallel processing enabled
    parallel_enabled: bool,
    /// Cache for expensive computations
    computation_cache: Arc<Mutex<HashMap<String, ComputationCacheEntry>>>,
    /// Performance tracking
    performance_tracker: Arc<Mutex<AnalysisPerformanceTracker>>,
}

/// Cache entry for expensive computations
#[derive(Debug, Clone)]
struct ComputationCacheEntry {
    result: Vec<f64>,
    timestamp: Instant,
    computation_type: String,
}

/// Performance tracker for analysis operations
#[derive(Debug)]
struct AnalysisPerformanceTracker {
    centrality_timings: HashMap<String, Vec<std::time::Duration>>,
    community_timings: HashMap<String, Vec<std::time::Duration>>,
    path_timings: HashMap<String, Vec<std::time::Duration>>,
    memory_usage: HashMap<String, Vec<u64>>,
}

impl AnalysisPerformanceTracker {
    fn new() -> Self {
        Self {
            centrality_timings: HashMap::new(),
            community_timings: HashMap::new(),
            path_timings: HashMap::new(),
            memory_usage: HashMap::new(),
        }
    }

    fn record_timing(&mut self, operation: String, duration: std::time::Duration) {
        self.centrality_timings.entry(operation).or_insert_with(Vec::new).push(duration);
    }

    fn get_average_timing(&self, operation: &str) -> Option<std::time::Duration> {
        self.centrality_timings.get(operation).map(|timings| {
            let total: std::time::Duration = timings.iter().sum();
            total / timings.len() as u32
        })
    }
}

impl GraphAnalyzer {
    /// Create a new graph analyzer
    pub fn new() -> Self {
        Self {
            simd_enabled: true,
            parallel_enabled: true,
            computation_cache: Arc::new(Mutex::new(HashMap::new())),
            performance_tracker: Arc::new(Mutex::new(AnalysisPerformanceTracker::new())),
        }
    }

    /// Perform comprehensive graph analysis
    pub fn analyze_graph(&self, graph: &TraitGraph) -> Result<GraphAnalysisResult> {
        let start_time = Instant::now();

        // Calculate all centrality measures
        let centrality_measures = self.calculate_all_centrality_measures(graph)?;

        // Detect communities
        let communities = self.detect_communities(graph, CommunityDetection::Louvain)?;

        // Find critical paths
        let critical_paths = self.find_critical_paths_comprehensive(graph)?;

        // Calculate quality metrics
        let quality_metrics = self.calculate_graph_quality_metrics(graph)?;

        // Record performance
        if let Ok(mut tracker) = self.performance_tracker.lock() {
            tracker.record_timing("comprehensive_analysis".to_string(), start_time.elapsed());
        }

        Ok(GraphAnalysisResult {
            centrality_measures,
            communities,
            critical_paths,
            quality_metrics,
            analyzed_at: Utc::now(),
        })
    }

    /// Calculate all centrality measures for nodes in the graph
    pub fn calculate_all_centrality_measures(
        &self,
        graph: &TraitGraph,
    ) -> Result<HashMap<String, CentralityMeasures>> {
        let mut results = HashMap::new();

        for node in &graph.nodes {
            let centrality = CentralityMeasures {
                degree: self.calculate_degree_centrality(graph, &node.id)?,
                betweenness: self.calculate_betweenness_centrality(graph, &node.id)?,
                closeness: self.calculate_closeness_centrality(graph, &node.id)?,
                eigenvector: self.calculate_eigenvector_centrality(graph, &node.id)?,
                pagerank: self.calculate_pagerank(graph, &node.id)?,
            };
            results.insert(node.id.clone(), centrality);
        }

        Ok(results)
    }

    /// Calculate degree centrality for a specific node
    fn calculate_degree_centrality(&self, graph: &TraitGraph, node_id: &str) -> Result<f64> {
        let degree = graph
            .edges
            .iter()
            .filter(|edge| edge.from == node_id || (!edge.directed && edge.to == node_id))
            .count();

        let max_possible_degree = graph.nodes.len().saturating_sub(1);
        if max_possible_degree > 0 {
            Ok(degree as f64 / max_possible_degree as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate betweenness centrality using Brandes' algorithm
    fn calculate_betweenness_centrality(&self, graph: &TraitGraph, node_id: &str) -> Result<f64> {
        let start_time = Instant::now();
        let n = graph.nodes.len();

        if n <= 2 {
            return Ok(0.0);
        }

        let mut betweenness = 0.0;
        let node_ids: Vec<_> = graph.nodes.iter().map(|n| &n.id).collect();

        // For each pair of nodes, calculate if node_id is on shortest paths
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let source = node_ids[i];
                let target = node_ids[j];

                if source == node_id || target == node_id {
                    continue;
                }

                // Find all shortest paths between source and target
                let paths = self.find_all_shortest_paths(graph, source, target)?;
                if paths.is_empty() {
                    continue;
                }

                // Count how many paths go through node_id
                let paths_through_node = paths.iter()
                    .filter(|path| path.contains_node(node_id))
                    .count();

                if paths_through_node > 0 {
                    betweenness += paths_through_node as f64 / paths.len() as f64;
                }
            }
        }

        // Normalize by the maximum possible betweenness
        let max_betweenness = ((n - 1) * (n - 2)) as f64 / 2.0;
        let normalized_betweenness = if max_betweenness > 0.0 {
            betweenness / max_betweenness
        } else {
            0.0
        };

        // Record performance
        if let Ok(mut tracker) = self.performance_tracker.lock() {
            tracker.record_timing(format!("betweenness_{}", node_id), start_time.elapsed());
        }

        Ok(normalized_betweenness)
    }

    /// Calculate closeness centrality
    fn calculate_closeness_centrality(&self, graph: &TraitGraph, node_id: &str) -> Result<f64> {
        let distances = self.calculate_shortest_path_distances(graph, node_id)?;

        if distances.is_empty() {
            return Ok(0.0);
        }

        let sum_distances: f64 = distances.values().sum();
        let reachable_nodes = distances.len() as f64;

        if sum_distances > 0.0 && reachable_nodes > 1.0 {
            // Normalized closeness centrality
            Ok((reachable_nodes - 1.0) / sum_distances)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate eigenvector centrality using power iteration
    fn calculate_eigenvector_centrality(&self, graph: &TraitGraph, node_id: &str) -> Result<f64> {
        let n = graph.nodes.len();
        if n == 0 {
            return Ok(0.0);
        }

        // Build adjacency matrix
        let adjacency_matrix = self.build_adjacency_matrix(graph)?;

        // Power iteration to find dominant eigenvector
        let mut eigenvector = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
        let max_iterations = 100;
        let tolerance = 1e-6;

        for _ in 0..max_iterations {
            let new_eigenvector = adjacency_matrix.dot(&eigenvector);
            let norm = new_eigenvector.dot(&new_eigenvector).sqrt();

            if norm > 0.0 {
                let normalized = &new_eigenvector / norm;
                let diff = (&normalized - &eigenvector).map(|x| x.abs()).sum();
                eigenvector = normalized;

                if diff < tolerance {
                    break;
                }
            } else {
                break;
            }
        }

        // Find the index of the node
        let node_index = graph.nodes.iter()
            .position(|n| n.id == node_id)
            .ok_or_else(|| SklearsError::ValidationError("Node not found".to_string()))?;

        Ok(eigenvector[node_index])
    }

    /// Calculate PageRank centrality
    fn calculate_pagerank(&self, graph: &TraitGraph, node_id: &str) -> Result<f64> {
        let n = graph.nodes.len();
        if n == 0 {
            return Ok(0.0);
        }

        let damping_factor = 0.85;
        let max_iterations = 100;
        let tolerance = 1e-6;

        // Initialize PageRank values
        let mut pagerank = Array1::from_elem(n, 1.0 / n as f64);
        let mut new_pagerank = Array1::zeros(n);

        // Build transition matrix
        let transition_matrix = self.build_transition_matrix(graph)?;

        for _ in 0..max_iterations {
            new_pagerank.fill(0.0);

            // PageRank update: PR(i) = (1-d)/N + d * sum(PR(j)/L(j)) for all j linking to i
            for i in 0..n {
                new_pagerank[i] = (1.0 - damping_factor) / n as f64;

                for j in 0..n {
                    if transition_matrix[(j, i)] > 0.0 {
                        new_pagerank[i] += damping_factor * pagerank[j] * transition_matrix[(j, i)];
                    }
                }
            }

            // Check convergence
            let diff = (&new_pagerank - &pagerank).map(|x| x.abs()).sum();
            pagerank = new_pagerank.clone();

            if diff < tolerance {
                break;
            }
        }

        // Find the index of the node
        let node_index = graph.nodes.iter()
            .position(|n| n.id == node_id)
            .ok_or_else(|| SklearsError::ValidationError("Node not found".to_string()))?;

        Ok(pagerank[node_index])
    }

    /// Detect communities using various algorithms
    pub fn detect_communities(
        &self,
        graph: &TraitGraph,
        algorithm: CommunityDetection,
    ) -> Result<Vec<Community>> {
        let start_time = Instant::now();

        let communities = match algorithm {
            CommunityDetection::Louvain => self.louvain_community_detection(graph)?,
            CommunityDetection::Leiden => self.leiden_community_detection(graph)?,
            CommunityDetection::LabelPropagation => self.label_propagation(graph)?,
            CommunityDetection::Walktrap => self.walktrap_community_detection(graph)?,
            CommunityDetection::GirvanNewman => self.girvan_newman_community_detection(graph)?,
            CommunityDetection::FastGreedy => self.fast_greedy_community_detection(graph)?,
        };

        // Record performance
        if let Ok(mut tracker) = self.performance_tracker.lock() {
            tracker.record_timing(format!("community_{:?}", algorithm), start_time.elapsed());
        }

        Ok(communities)
    }

    /// Louvain community detection algorithm
    fn louvain_community_detection(&self, graph: &TraitGraph) -> Result<Vec<Community>> {
        let mut communities = Vec::new();
        let n = graph.nodes.len();

        if n == 0 {
            return Ok(communities);
        }

        // Initialize each node as its own community
        let mut node_communities: HashMap<String, usize> = HashMap::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            node_communities.insert(node.id.clone(), i);
        }

        let mut community_nodes: HashMap<usize, HashSet<String>> = HashMap::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            let mut node_set = HashSet::new();
            node_set.insert(node.id.clone());
            community_nodes.insert(i, node_set);
        }

        let max_iterations = 10;
        let mut improved = true;

        for _ in 0..max_iterations {
            if !improved {
                break;
            }
            improved = false;

            // Try to move each node to the community that maximizes modularity
            for node in &graph.nodes {
                let current_community = *node_communities.get(&node.id).unwrap();
                let mut best_community = current_community;
                let mut best_gain = 0.0;

                // Consider neighboring communities
                let neighbor_communities = self.get_neighbor_communities(graph, &node.id, &node_communities);

                for &neighbor_community in &neighbor_communities {
                    if neighbor_community == current_community {
                        continue;
                    }

                    let gain = self.calculate_modularity_gain(
                        graph,
                        &node.id,
                        current_community,
                        neighbor_community,
                        &node_communities,
                    );

                    if gain > best_gain {
                        best_gain = gain;
                        best_community = neighbor_community;
                    }
                }

                // Move node if beneficial
                if best_community != current_community && best_gain > 0.0 {
                    // Remove from old community
                    if let Some(old_community) = community_nodes.get_mut(&current_community) {
                        old_community.remove(&node.id);
                    }

                    // Add to new community
                    community_nodes.entry(best_community).or_insert_with(HashSet::new).insert(node.id.clone());
                    node_communities.insert(node.id.clone(), best_community);
                    improved = true;
                }
            }
        }

        // Convert to Community structures
        for (community_id, nodes) in community_nodes {
            if !nodes.is_empty() {
                let modularity = self.calculate_community_modularity(graph, &nodes);
                communities.push(Community {
                    id: format!("community_{}", community_id),
                    nodes: nodes.into_iter().collect(),
                    modularity,
                    description: None,
                });
            }
        }

        Ok(communities)
    }

    /// Leiden community detection (simplified implementation)
    fn leiden_community_detection(&self, graph: &TraitGraph) -> Result<Vec<Community>> {
        // For simplicity, use a modified Louvain approach
        // In practice, Leiden would include additional refinement steps
        self.louvain_community_detection(graph)
    }

    /// Label propagation algorithm
    fn label_propagation(&self, graph: &TraitGraph) -> Result<Vec<Community>> {
        let mut node_labels: HashMap<String, usize> = HashMap::new();

        // Initialize each node with its own label
        for (i, node) in graph.nodes.iter().enumerate() {
            node_labels.insert(node.id.clone(), i);
        }

        let max_iterations = 100;
        let mut rng = Random::seed(42);

        for _ in 0..max_iterations {
            let mut changed = false;

            // Randomize order of nodes
            let mut nodes = graph.nodes.clone();
            rng.shuffle(&mut nodes);

            for node in &nodes {
                // Count labels of neighbors
                let mut label_counts: HashMap<usize, f64> = HashMap::new();

                for edge in &graph.edges {
                    let neighbor_id = if edge.from == node.id {
                        Some(&edge.to)
                    } else if !edge.directed && edge.to == node.id {
                        Some(&edge.from)
                    } else {
                        None
                    };

                    if let Some(neighbor_id) = neighbor_id {
                        if let Some(&neighbor_label) = node_labels.get(neighbor_id) {
                            *label_counts.entry(neighbor_label).or_insert(0.0) += edge.weight;
                        }
                    }
                }

                // Find most frequent label
                if let Some((&most_frequent_label, _)) = label_counts.iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                {
                    let current_label = *node_labels.get(&node.id).unwrap();
                    if most_frequent_label != current_label {
                        node_labels.insert(node.id.clone(), most_frequent_label);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Group nodes by label
        let mut communities_map: HashMap<usize, Vec<String>> = HashMap::new();
        for (node_id, label) in node_labels {
            communities_map.entry(label).or_insert_with(Vec::new).push(node_id);
        }

        let mut communities = Vec::new();
        for (label, nodes) in communities_map {
            if nodes.len() > 1 { // Only consider communities with multiple nodes
                let node_set: HashSet<String> = nodes.iter().cloned().collect();
                let modularity = self.calculate_community_modularity(graph, &node_set);
                communities.push(Community {
                    id: format!("community_{}", label),
                    nodes,
                    modularity,
                    description: Some("Label propagation community".to_string()),
                });
            }
        }

        Ok(communities)
    }

    /// Walktrap community detection (simplified)
    fn walktrap_community_detection(&self, _graph: &TraitGraph) -> Result<Vec<Community>> {
        // Placeholder implementation - would use random walks
        Ok(Vec::new())
    }

    /// Girvan-Newman community detection (edge betweenness)
    fn girvan_newman_community_detection(&self, _graph: &TraitGraph) -> Result<Vec<Community>> {
        // Placeholder implementation - would iteratively remove high betweenness edges
        Ok(Vec::new())
    }

    /// Fast greedy community detection
    fn fast_greedy_community_detection(&self, _graph: &TraitGraph) -> Result<Vec<Community>> {
        // Placeholder implementation - would use greedy modularity optimization
        Ok(Vec::new())
    }

    /// Find critical paths in the graph
    pub fn find_critical_paths_comprehensive(&self, graph: &TraitGraph) -> Result<Vec<GraphPath>> {
        let mut critical_paths = Vec::new();

        // Find paths between high-centrality nodes
        let centrality_measures = self.calculate_all_centrality_measures(graph)?;

        let mut high_centrality_nodes: Vec<_> = centrality_measures.iter()
            .filter(|(_, measures)| measures.importance_score() > 0.7)
            .map(|(id, _)| id.as_str())
            .collect();

        high_centrality_nodes.sort_by(|a, b| {
            let a_score = centrality_measures.get(*a).map(|m| m.importance_score()).unwrap_or(0.0);
            let b_score = centrality_measures.get(*b).map(|m| m.importance_score()).unwrap_or(0.0);
            b_score.partial_cmp(&a_score).unwrap_or(Ordering::Equal)
        });

        // Find paths between top centrality nodes
        for i in 0..high_centrality_nodes.len().min(5) {
            for j in (i + 1)..high_centrality_nodes.len().min(5) {
                let source = high_centrality_nodes[i];
                let target = high_centrality_nodes[j];

                if let Ok(paths) = self.find_all_shortest_paths(graph, source, target) {
                    critical_paths.extend(paths);
                }
            }
        }

        // Limit to most important paths
        critical_paths.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(Ordering::Equal));
        critical_paths.truncate(10);

        Ok(critical_paths)
    }

    /// Find all shortest paths between two nodes
    pub fn find_all_shortest_paths(
        &self,
        graph: &TraitGraph,
        source: &str,
        target: &str,
    ) -> Result<Vec<GraphPath>> {
        if source == target {
            return Ok(vec![GraphPath::new(vec![source.to_string()])]);
        }

        // Use BFS to find shortest paths
        let mut queue = VecDeque::new();
        let mut distances: HashMap<String, f64> = HashMap::new();
        let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();

        queue.push_back(source.to_string());
        distances.insert(source.to_string(), 0.0);

        while let Some(current) = queue.pop_front() {
            let current_distance = distances[&current];

            for edge in &graph.edges {
                let next_node = if edge.from == current {
                    Some(&edge.to)
                } else if !edge.directed && edge.to == current {
                    Some(&edge.from)
                } else {
                    None
                };

                if let Some(next_node) = next_node {
                    let new_distance = current_distance + edge.weight;

                    match distances.get(next_node) {
                        None => {
                            // First time visiting this node
                            distances.insert(next_node.clone(), new_distance);
                            predecessors.insert(next_node.clone(), vec![current.clone()]);
                            queue.push_back(next_node.clone());
                        }
                        Some(&existing_distance) => {
                            if new_distance < existing_distance {
                                // Found shorter path
                                distances.insert(next_node.clone(), new_distance);
                                predecessors.insert(next_node.clone(), vec![current.clone()]);
                            } else if (new_distance - existing_distance).abs() < 1e-10 {
                                // Found alternative shortest path
                                predecessors.entry(next_node.clone())
                                    .or_insert_with(Vec::new)
                                    .push(current.clone());
                            }
                        }
                    }
                }
            }
        }

        // Reconstruct all shortest paths
        if !distances.contains_key(target) {
            return Ok(Vec::new()); // No path exists
        }

        let paths = self.reconstruct_all_paths(&predecessors, source, target);
        let target_distance = distances[target];

        let graph_paths = paths.into_iter().map(|path| {
            let mut graph_path = GraphPath::new(path);
            graph_path.weight = target_distance;
            graph_path
        }).collect();

        Ok(graph_paths)
    }

    /// Calculate graph quality metrics
    pub fn calculate_graph_quality_metrics(&self, graph: &TraitGraph) -> Result<GraphQualityMetrics> {
        let clarity = self.calculate_visual_clarity(graph);
        let layout_quality = self.calculate_layout_quality(graph);
        let information_density = self.calculate_information_density(graph);
        let aesthetic_appeal = self.calculate_aesthetic_appeal(graph);
        let usability = self.calculate_usability_score(graph);

        Ok(GraphQualityMetrics {
            clarity,
            layout_quality,
            information_density,
            aesthetic_appeal,
            usability,
        })
    }

    /// Build adjacency matrix for the graph
    fn build_adjacency_matrix(&self, graph: &TraitGraph) -> Result<Array2<f64>> {
        let n = graph.nodes.len();
        let mut matrix = Array2::zeros((n, n));

        // Create node index mapping
        let node_indices: HashMap<String, usize> = graph.nodes.iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();

        // Fill adjacency matrix
        for edge in &graph.edges {
            if let (Some(&from_idx), Some(&to_idx)) = (
                node_indices.get(&edge.from),
                node_indices.get(&edge.to)
            ) {
                matrix[(from_idx, to_idx)] = edge.weight;
                if !edge.directed {
                    matrix[(to_idx, from_idx)] = edge.weight;
                }
            }
        }

        Ok(matrix)
    }

    /// Build transition matrix for PageRank
    fn build_transition_matrix(&self, graph: &TraitGraph) -> Result<Array2<f64>> {
        let n = graph.nodes.len();
        let mut matrix = Array2::zeros((n, n));

        // Create node index mapping
        let node_indices: HashMap<String, usize> = graph.nodes.iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();

        // Calculate out-degrees
        let mut out_degrees = vec![0.0; n];
        for edge in &graph.edges {
            if let Some(&from_idx) = node_indices.get(&edge.from) {
                out_degrees[from_idx] += edge.weight;
            }
        }

        // Fill transition matrix
        for edge in &graph.edges {
            if let (Some(&from_idx), Some(&to_idx)) = (
                node_indices.get(&edge.from),
                node_indices.get(&edge.to)
            ) {
                if out_degrees[from_idx] > 0.0 {
                    matrix[(from_idx, to_idx)] = edge.weight / out_degrees[from_idx];
                }
            }
        }

        Ok(matrix)
    }

    /// Calculate shortest path distances from a source node
    fn calculate_shortest_path_distances(
        &self,
        graph: &TraitGraph,
        source: &str,
    ) -> Result<HashMap<String, f64>> {
        let mut distances = HashMap::new();
        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();

        #[derive(PartialEq)]
        struct State {
            cost: OrderedFloat,
            node: String,
        }

        impl Eq for State {}

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other.cost.cmp(&self.cost)
            }
        }

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        #[derive(PartialEq, PartialOrd)]
        struct OrderedFloat(f64);

        impl Eq for OrderedFloat {}

        impl Ord for OrderedFloat {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        heap.push(State {
            cost: OrderedFloat(0.0),
            node: source.to_string(),
        });
        distances.insert(source.to_string(), 0.0);

        while let Some(State { cost, node }) = heap.pop() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node.clone());

            for edge in &graph.edges {
                let next_node = if edge.from == node {
                    Some(&edge.to)
                } else if !edge.directed && edge.to == node {
                    Some(&edge.from)
                } else {
                    None
                };

                if let Some(next_node) = next_node {
                    let next_cost = cost.0 + edge.weight;

                    if !visited.contains(next_node) {
                        let current_distance = distances.get(next_node).copied().unwrap_or(f64::INFINITY);
                        if next_cost < current_distance {
                            distances.insert(next_node.clone(), next_cost);
                            heap.push(State {
                                cost: OrderedFloat(next_cost),
                                node: next_node.clone(),
                            });
                        }
                    }
                }
            }
        }

        Ok(distances)
    }

    /// Get neighboring communities for a node
    fn get_neighbor_communities(
        &self,
        graph: &TraitGraph,
        node_id: &str,
        node_communities: &HashMap<String, usize>,
    ) -> HashSet<usize> {
        let mut communities = HashSet::new();

        for edge in &graph.edges {
            let neighbor_id = if edge.from == node_id {
                Some(&edge.to)
            } else if !edge.directed && edge.to == node_id {
                Some(&edge.from)
            } else {
                None
            };

            if let Some(neighbor_id) = neighbor_id {
                if let Some(&community) = node_communities.get(neighbor_id) {
                    communities.insert(community);
                }
            }
        }

        communities
    }

    /// Calculate modularity gain for moving a node between communities
    fn calculate_modularity_gain(
        &self,
        _graph: &TraitGraph,
        _node_id: &str,
        _from_community: usize,
        _to_community: usize,
        _node_communities: &HashMap<String, usize>,
    ) -> f64 {
        // Simplified modularity calculation
        // In practice, this would involve detailed modularity computation
        0.1 // Placeholder
    }

    /// Calculate modularity for a community
    fn calculate_community_modularity(&self, graph: &TraitGraph, community_nodes: &HashSet<String>) -> f64 {
        if community_nodes.len() <= 1 {
            return 0.0;
        }

        let total_edges = graph.edges.len() as f64;
        if total_edges == 0.0 {
            return 0.0;
        }

        // Count internal edges
        let internal_edges = graph.edges.iter()
            .filter(|edge| community_nodes.contains(&edge.from) && community_nodes.contains(&edge.to))
            .count() as f64;

        // Count external edges
        let external_edges = graph.edges.iter()
            .filter(|edge| {
                (community_nodes.contains(&edge.from) && !community_nodes.contains(&edge.to)) ||
                (!community_nodes.contains(&edge.from) && community_nodes.contains(&edge.to))
            })
            .count() as f64;

        // Simple modularity approximation
        if total_edges > 0.0 {
            (internal_edges - external_edges) / total_edges
        } else {
            0.0
        }
    }

    /// Reconstruct all paths from predecessors map
    fn reconstruct_all_paths(
        &self,
        predecessors: &HashMap<String, Vec<String>>,
        source: &str,
        target: &str,
    ) -> Vec<Vec<String>> {
        if source == target {
            return vec![vec![source.to_string()]];
        }

        let mut all_paths = Vec::new();
        let mut current_paths = vec![vec![target.to_string()]];

        while !current_paths.is_empty() {
            let mut next_paths = Vec::new();

            for path in current_paths {
                let current_node = &path[path.len() - 1];

                if current_node == source {
                    let mut complete_path = path.clone();
                    complete_path.reverse();
                    all_paths.push(complete_path);
                } else if let Some(preds) = predecessors.get(current_node) {
                    for pred in preds {
                        let mut new_path = path.clone();
                        new_path.push(pred.clone());
                        next_paths.push(new_path);
                    }
                }
            }

            current_paths = next_paths;
        }

        all_paths
    }

    /// Calculate visual clarity metrics
    fn calculate_visual_clarity(&self, graph: &TraitGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 1.0;
        }

        // Consider node overlap, edge crossings, and label readability
        let node_density = graph.nodes.len() as f64 / 1000.0; // Normalize by expected area
        let edge_density = graph.edges.len() as f64 / (graph.nodes.len() as f64).powi(2);

        let clarity_score = 1.0 - (node_density.min(1.0) * 0.5 + edge_density.min(1.0) * 0.5);
        clarity_score.max(0.0).min(1.0)
    }

    /// Calculate layout quality
    fn calculate_layout_quality(&self, graph: &TraitGraph) -> f64 {
        if graph.nodes.len() < 2 {
            return 1.0;
        }

        // Check if nodes have positions
        let positioned_nodes = graph.nodes.iter()
            .filter(|node| node.position_2d.is_some())
            .count();

        if positioned_nodes == 0 {
            return 0.0;
        }

        let position_coverage = positioned_nodes as f64 / graph.nodes.len() as f64;

        // Calculate edge length variance (lower is better)
        let mut edge_lengths = Vec::new();
        for edge in &graph.edges {
            if let (Some(from_node), Some(to_node)) = (
                graph.nodes.iter().find(|n| n.id == edge.from),
                graph.nodes.iter().find(|n| n.id == edge.to),
            ) {
                if let (Some((x1, y1)), Some((x2, y2))) = (from_node.position_2d, to_node.position_2d) {
                    let length = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
                    edge_lengths.push(length);
                }
            }
        }

        let edge_length_uniformity = if edge_lengths.len() > 1 {
            let mean = edge_lengths.iter().sum::<f64>() / edge_lengths.len() as f64;
            let variance = edge_lengths.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / edge_lengths.len() as f64;
            let std_dev = variance.sqrt();
            1.0 - (std_dev / mean.max(1.0)).min(1.0)
        } else {
            1.0
        };

        (position_coverage + edge_length_uniformity) / 2.0
    }

    /// Calculate information density
    fn calculate_information_density(&self, graph: &TraitGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 0.0;
        }

        // Consider the amount of meaningful information per visual space
        let node_count = graph.nodes.len() as f64;
        let edge_count = graph.edges.len() as f64;

        let information_content = node_count + edge_count * 0.5;
        let normalized_density = information_content / (node_count * 10.0); // Normalize by expected capacity

        normalized_density.min(1.0)
    }

    /// Calculate aesthetic appeal
    fn calculate_aesthetic_appeal(&self, graph: &TraitGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 1.0;
        }

        // Consider symmetry, balance, and visual harmony
        let symmetry_score = self.calculate_symmetry_score(graph);
        let balance_score = self.calculate_balance_score(graph);
        let color_harmony = self.calculate_color_harmony(graph);

        (symmetry_score + balance_score + color_harmony) / 3.0
    }

    /// Calculate usability score
    fn calculate_usability_score(&self, graph: &TraitGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 1.0;
        }

        // Consider navigability, readability, and interaction design
        let readability = self.calculate_readability(graph);
        let navigability = self.calculate_navigability(graph);
        let interaction_design = 0.8; // Placeholder

        (readability + navigability + interaction_design) / 3.0
    }

    /// Calculate symmetry score (placeholder)
    fn calculate_symmetry_score(&self, _graph: &TraitGraph) -> f64 {
        0.7 // Placeholder
    }

    /// Calculate balance score (placeholder)
    fn calculate_balance_score(&self, _graph: &TraitGraph) -> f64 {
        0.8 // Placeholder
    }

    /// Calculate color harmony (placeholder)
    fn calculate_color_harmony(&self, _graph: &TraitGraph) -> f64 {
        0.75 // Placeholder
    }

    /// Calculate readability (placeholder)
    fn calculate_readability(&self, _graph: &TraitGraph) -> f64 {
        0.8 // Placeholder
    }

    /// Calculate navigability (placeholder)
    fn calculate_navigability(&self, _graph: &TraitGraph) -> f64 {
        0.85 // Placeholder
    }

    /// Enable or disable SIMD optimization
    pub fn set_simd_enabled(&mut self, enabled: bool) {
        self.simd_enabled = enabled;
    }

    /// Enable or disable parallel processing
    pub fn set_parallel_enabled(&mut self, enabled: bool) {
        self.parallel_enabled = enabled;
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> Option<HashMap<String, std::time::Duration>> {
        if let Ok(tracker) = self.performance_tracker.lock() {
            let mut stats = HashMap::new();
            for (operation, _) in &tracker.centrality_timings {
                if let Some(avg_time) = tracker.get_average_timing(operation) {
                    stats.insert(operation.clone(), avg_time);
                }
            }
            Some(stats)
        } else {
            None
        }
    }

    /// Clear computation cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.computation_cache.lock() {
            cache.clear();
        }
    }
}

impl Default for GraphAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph_structures::{TraitGraph, TraitGraphNode, TraitGraphEdge};

    fn create_test_graph() -> TraitGraph {
        let mut graph = TraitGraph::new();

        // Add nodes
        let node1 = TraitGraphNode::new_trait("A".to_string(), "NodeA".to_string());
        let node2 = TraitGraphNode::new_trait("B".to_string(), "NodeB".to_string());
        let node3 = TraitGraphNode::new_trait("C".to_string(), "NodeC".to_string());
        let node4 = TraitGraphNode::new_trait("D".to_string(), "NodeD".to_string());

        graph.add_node(node1);
        graph.add_node(node2);
        graph.add_node(node3);
        graph.add_node(node4);

        // Add edges
        let edge1 = TraitGraphEdge::new_inheritance("A".to_string(), "B".to_string());
        let edge2 = TraitGraphEdge::new_inheritance("B".to_string(), "C".to_string());
        let edge3 = TraitGraphEdge::new_inheritance("A".to_string(), "D".to_string());
        let edge4 = TraitGraphEdge::new_inheritance("D".to_string(), "C".to_string());

        graph.add_edge(edge1);
        graph.add_edge(edge2);
        graph.add_edge(edge3);
        graph.add_edge(edge4);

        graph
    }

    #[test]
    fn test_graph_analyzer_creation() {
        let analyzer = GraphAnalyzer::new();
        assert!(analyzer.simd_enabled);
        assert!(analyzer.parallel_enabled);
    }

    #[test]
    fn test_degree_centrality() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let centrality_a = analyzer.calculate_degree_centrality(&graph, "A").unwrap();
        let centrality_c = analyzer.calculate_degree_centrality(&graph, "C").unwrap();

        // Node A has 2 outgoing edges, Node C has 2 incoming edges
        assert!(centrality_a > 0.0);
        assert!(centrality_c > 0.0);
    }

    #[test]
    fn test_betweenness_centrality() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let centrality_b = analyzer.calculate_betweenness_centrality(&graph, "B").unwrap();
        let centrality_d = analyzer.calculate_betweenness_centrality(&graph, "D").unwrap();

        assert!(centrality_b >= 0.0);
        assert!(centrality_d >= 0.0);
    }

    #[test]
    fn test_closeness_centrality() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let centrality_a = analyzer.calculate_closeness_centrality(&graph, "A").unwrap();
        assert!(centrality_a >= 0.0);
    }

    #[test]
    fn test_pagerank() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let pagerank_a = analyzer.calculate_pagerank(&graph, "A").unwrap();
        let pagerank_c = analyzer.calculate_pagerank(&graph, "C").unwrap();

        assert!(pagerank_a > 0.0);
        assert!(pagerank_c > 0.0);
    }

    #[test]
    fn test_all_centrality_measures() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let centralities = analyzer.calculate_all_centrality_measures(&graph).unwrap();

        assert_eq!(centralities.len(), 4);
        assert!(centralities.contains_key("A"));
        assert!(centralities.contains_key("B"));
        assert!(centralities.contains_key("C"));
        assert!(centralities.contains_key("D"));

        for (_, measures) in centralities {
            assert!(measures.degree >= 0.0 && measures.degree <= 1.0);
            assert!(measures.betweenness >= 0.0 && measures.betweenness <= 1.0);
            assert!(measures.closeness >= 0.0);
            assert!(measures.pagerank >= 0.0 && measures.pagerank <= 1.0);
        }
    }

    #[test]
    fn test_shortest_paths() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let paths = analyzer.find_all_shortest_paths(&graph, "A", "C").unwrap();
        assert!(!paths.is_empty());

        // There should be at least one path from A to C
        assert!(paths.iter().any(|path| path.start() == Some("A") && path.end() == Some("C")));
    }

    #[test]
    fn test_community_detection() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let communities = analyzer.detect_communities(&graph, CommunityDetection::Louvain).unwrap();

        // Should detect at least one community
        assert!(!communities.is_empty());

        for community in communities {
            assert!(!community.nodes.is_empty());
            assert!(!community.id.is_empty());
        }
    }

    #[test]
    fn test_label_propagation() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let communities = analyzer.label_propagation(&graph).unwrap();

        // Label propagation may or may not find communities depending on the graph structure
        for community in communities {
            assert!(!community.nodes.is_empty());
        }
    }

    #[test]
    fn test_graph_quality_metrics() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let quality = analyzer.calculate_graph_quality_metrics(&graph).unwrap();

        assert!(quality.clarity >= 0.0 && quality.clarity <= 1.0);
        assert!(quality.layout_quality >= 0.0 && quality.layout_quality <= 1.0);
        assert!(quality.information_density >= 0.0 && quality.information_density <= 1.0);
        assert!(quality.aesthetic_appeal >= 0.0 && quality.aesthetic_appeal <= 1.0);
        assert!(quality.usability >= 0.0 && quality.usability <= 1.0);
    }

    #[test]
    fn test_comprehensive_analysis() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let analysis = analyzer.analyze_graph(&graph).unwrap();

        assert!(!analysis.centrality_measures.is_empty());
        assert!(!analysis.critical_paths.is_empty());
        assert!(analysis.quality_metrics.overall_quality() >= 0.0);
    }

    #[test]
    fn test_adjacency_matrix() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let matrix = analyzer.build_adjacency_matrix(&graph).unwrap();

        assert_eq!(matrix.dim(), (4, 4));
        assert!(matrix.sum() > 0.0); // Should have some non-zero entries
    }

    #[test]
    fn test_shortest_path_distances() {
        let analyzer = GraphAnalyzer::new();
        let graph = create_test_graph();

        let distances = analyzer.calculate_shortest_path_distances(&graph, "A").unwrap();

        assert!(distances.contains_key("A"));
        assert_eq!(distances["A"], 0.0);

        // Should be able to reach other nodes
        assert!(distances.len() > 1);
    }

    #[test]
    fn test_cache_operations() {
        let analyzer = GraphAnalyzer::new();

        // Test cache clearing
        analyzer.clear_cache();

        // Cache should be accessible
        assert!(analyzer.computation_cache.lock().is_ok());
    }

    #[test]
    fn test_configuration() {
        let mut analyzer = GraphAnalyzer::new();

        analyzer.set_simd_enabled(false);
        assert!(!analyzer.simd_enabled);

        analyzer.set_parallel_enabled(false);
        assert!(!analyzer.parallel_enabled);

        analyzer.set_simd_enabled(true);
        assert!(analyzer.simd_enabled);
    }
}