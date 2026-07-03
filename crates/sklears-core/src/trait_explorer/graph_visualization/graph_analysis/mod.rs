// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

use super::graph_config::CommunityDetection;
use super::graph_structures::{
    TraitGraph, CentralityMeasures, Community, GraphPath, GraphAnalysisResult,
    GraphQualityMetrics,
};
use crate::error::{Result, SklearsError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;
use chrono::Utc;
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use std::f64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
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
    /// Record a timing sample, bucketed by the operation name's prefix
    /// (`"community_*"` and `"path_*"`/`"shortest_path*"` route to their
    /// respective maps; everything else — centrality computations and the
    /// overall `"comprehensive_analysis"` timing — is treated as a
    /// centrality-class timing) so [`Self::get_average_timing`] and
    /// [`GraphAnalyzer::get_performance_stats`] can report each category.
    fn record_timing(&mut self, operation: String, duration: std::time::Duration) {
        let bucket = if operation.starts_with("community_") {
            &mut self.community_timings
        } else if operation.starts_with("path_")
            || operation.starts_with("shortest_path")
        {
            &mut self.path_timings
        } else {
            &mut self.centrality_timings
        };
        bucket.entry(operation).or_default().push(duration);
    }
    /// Record a memory-usage sample (in bytes) for `operation`.
    fn record_memory_usage(&mut self, operation: String, bytes: u64) {
        self.memory_usage.entry(operation).or_default().push(bytes);
    }
    /// Look up the average recorded duration for `operation`, searching all
    /// three timing categories.
    fn get_average_timing(&self, operation: &str) -> Option<std::time::Duration> {
        for bucket in [
            &self.centrality_timings,
            &self.community_timings,
            &self.path_timings,
        ] {
            if let Some(timings) = bucket.get(operation) {
                let total: std::time::Duration = timings.iter().sum();
                return Some(total / timings.len() as u32);
            }
        }
        None
    }
    /// Look up the average recorded memory usage (in bytes) for `operation`.
    fn get_average_memory_usage(&self, operation: &str) -> Option<u64> {
        self.memory_usage
            .get(operation)
            .filter(|samples| !samples.is_empty())
            .map(|samples| samples.iter().sum::<u64>() / samples.len() as u64)
    }
    /// Iterate over every operation name that has recorded timings, across
    /// all three categories.
    fn timed_operations(&self) -> impl Iterator<Item = &String> {
        self.centrality_timings
            .keys()
            .chain(self.community_timings.keys())
            .chain(self.path_timings.keys())
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
        let centrality_measures = self.calculate_all_centrality_measures(graph)?;
        let communities = self.detect_communities(graph, CommunityDetection::Louvain)?;
        let critical_paths = self.find_critical_paths_comprehensive(graph)?;
        let quality_metrics = self.calculate_graph_quality_metrics(graph)?;
        if let Ok(mut tracker) = self.performance_tracker.lock() {
            tracker
                .record_timing(
                    "comprehensive_analysis".to_string(),
                    start_time.elapsed(),
                );
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
    /// Calculate degree centrality for a specific node.
    ///
    /// Counts every edge that touches `node_id` as either endpoint,
    /// regardless of its `directed` flag: degree centrality is a measure of
    /// how structurally connected a node is (how many relationships it
    /// participates in), which for a directed edge includes both being the
    /// source (e.g. a trait's supertrait) and being the target (e.g. a
    /// trait's implementation) — unlike path-following analyses, there is
    /// no reason to ignore incoming edges here.
    fn calculate_degree_centrality(
        &self,
        graph: &TraitGraph,
        node_id: &str,
    ) -> Result<f64> {
        let degree = graph
            .edges
            .iter()
            .filter(|edge| edge.from == node_id || edge.to == node_id)
            .count();
        let max_possible_degree = graph.nodes.len().saturating_sub(1);
        if max_possible_degree > 0 {
            Ok(degree as f64 / max_possible_degree as f64)
        } else {
            Ok(0.0)
        }
    }
    /// Calculate betweenness centrality using Brandes' algorithm
    ///
    /// This is the most expensive centrality measure (all-pairs shortest
    /// paths), so results are cached in `self.computation_cache` keyed by a
    /// structural hash of the graph plus the node id; repeated calls for the
    /// same graph (e.g. once per node from
    /// [`Self::calculate_all_centrality_measures`], or across successive
    /// [`Self::analyze_graph`] calls on an unchanged graph) reuse the cached
    /// result instead of recomputing it, as long as the entry has not
    /// exceeded `CACHE_TTL`.
    fn calculate_betweenness_centrality(
        &self,
        graph: &TraitGraph,
        node_id: &str,
    ) -> Result<f64> {
        const CACHE_TTL: Duration = Duration::from_secs(300);
        const COMPUTATION_TYPE: &str = "betweenness";
        let cache_key = format!(
            "{}:{}:{}", COMPUTATION_TYPE, Self::graph_structural_hash(graph), node_id
        );
        if let Ok(cache) = self.computation_cache.lock() {
            if let Some(entry) = cache.get(&cache_key) {
                if entry.computation_type == COMPUTATION_TYPE
                    && entry.timestamp.elapsed() < CACHE_TTL
                {
                    if let Some(&cached) = entry.result.first() {
                        return Ok(cached);
                    }
                }
            }
        }
        let start_time = Instant::now();
        let n = graph.nodes.len();
        if n <= 2 {
            return Ok(0.0);
        }
        let mut betweenness = 0.0;
        let node_ids: Vec<_> = graph.nodes.iter().map(|n| &n.id).collect();
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let source = node_ids[i];
                let target = node_ids[j];
                if source == node_id || target == node_id {
                    continue;
                }
                let paths = self.find_all_shortest_paths(graph, source, target)?;
                if paths.is_empty() {
                    continue;
                }
                let paths_through_node = paths
                    .iter()
                    .filter(|path| path.contains_node(node_id))
                    .count();
                if paths_through_node > 0 {
                    betweenness += paths_through_node as f64 / paths.len() as f64;
                }
            }
        }
        let max_betweenness = ((n - 1) * (n - 2)) as f64 / 2.0;
        let normalized_betweenness = if max_betweenness > 0.0 {
            betweenness / max_betweenness
        } else {
            0.0
        };
        if let Ok(mut cache) = self.computation_cache.lock() {
            cache
                .insert(
                    cache_key,
                    ComputationCacheEntry {
                        result: vec![normalized_betweenness],
                        timestamp: Instant::now(),
                        computation_type: COMPUTATION_TYPE.to_string(),
                    },
                );
        }
        if let Ok(mut tracker) = self.performance_tracker.lock() {
            tracker
                .record_timing(format!("betweenness_{}", node_id), start_time.elapsed());
            let estimated_bytes = (n * std::mem::size_of::<CentralityMeasures>()) as u64;
            tracker.record_memory_usage("betweenness".to_string(), estimated_bytes);
        }
        Ok(normalized_betweenness)
    }
    /// Compute a structural hash of the graph (node ids, and each edge's
    /// endpoints and directedness) for use as part of a cache key: two
    /// calls with graphs that have the same nodes and edges hash equal,
    /// while any structural change (added/removed/rewired node or edge)
    /// changes the hash, invalidating stale cache entries.
    fn graph_structural_hash(graph: &TraitGraph) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut node_ids: Vec<&str> = graph
            .nodes
            .iter()
            .map(|n| n.id.as_str())
            .collect();
        node_ids.sort_unstable();
        let mut edge_keys: Vec<(&str, &str, bool)> = graph
            .edges
            .iter()
            .map(|e| (e.from.as_str(), e.to.as_str(), e.directed))
            .collect();
        edge_keys.sort_unstable();
        let mut hasher = DefaultHasher::new();
        node_ids.hash(&mut hasher);
        edge_keys.hash(&mut hasher);
        hasher.finish()
    }
    /// Calculate closeness centrality
    fn calculate_closeness_centrality(
        &self,
        graph: &TraitGraph,
        node_id: &str,
    ) -> Result<f64> {
        let distances = self.calculate_shortest_path_distances(graph, node_id)?;
        if distances.is_empty() {
            return Ok(0.0);
        }
        let sum_distances: f64 = distances.values().sum();
        let reachable_nodes = distances.len() as f64;
        if sum_distances > 0.0 && reachable_nodes > 1.0 {
            Ok((reachable_nodes - 1.0) / sum_distances)
        } else {
            Ok(0.0)
        }
    }
    /// Calculate eigenvector centrality using power iteration
    fn calculate_eigenvector_centrality(
        &self,
        graph: &TraitGraph,
        node_id: &str,
    ) -> Result<f64> {
        let n = graph.nodes.len();
        if n == 0 {
            return Ok(0.0);
        }
        let adjacency_matrix = self.build_adjacency_matrix(graph)?;
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
        let node_index = graph
            .nodes
            .iter()
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
        let mut pagerank = Array1::from_elem(n, 1.0 / n as f64);
        let mut new_pagerank = Array1::zeros(n);
        let transition_matrix = self.build_transition_matrix(graph)?;
        for _ in 0..max_iterations {
            new_pagerank.fill(0.0);
            for i in 0..n {
                new_pagerank[i] = (1.0 - damping_factor) / n as f64;
                for j in 0..n {
                    if transition_matrix[(j, i)] > 0.0 {
                        new_pagerank[i]
                            += damping_factor * pagerank[j] * transition_matrix[(j, i)];
                    }
                }
            }
            let diff = (&new_pagerank - &pagerank).map(|x| x.abs()).sum();
            pagerank = new_pagerank.clone();
            if diff < tolerance {
                break;
            }
        }
        let node_index = graph
            .nodes
            .iter()
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
            CommunityDetection::GirvanNewman => {
                self.girvan_newman_community_detection(graph)?
            }
            CommunityDetection::FastGreedy => {
                self.fast_greedy_community_detection(graph)?
            }
        };
        if let Ok(mut tracker) = self.performance_tracker.lock() {
            tracker
                .record_timing(
                    format!("community_{:?}", algorithm),
                    start_time.elapsed(),
                );
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
            for node in &graph.nodes {
                let current_community = *node_communities
                    .get(&node.id)
                    .expect("get should succeed");
                let mut best_community = current_community;
                let mut best_gain = 0.0;
                let neighbor_communities = self
                    .get_neighbor_communities(graph, &node.id, &node_communities);
                for &neighbor_community in &neighbor_communities {
                    if neighbor_community == current_community {
                        continue;
                    }
                    let gain = self
                        .calculate_modularity_gain(
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
                if best_community != current_community && best_gain > 0.0 {
                    if let Some(old_community) = community_nodes
                        .get_mut(&current_community)
                    {
                        old_community.remove(&node.id);
                    }
                    community_nodes
                        .entry(best_community)
                        .or_default()
                        .insert(node.id.clone());
                    node_communities.insert(node.id.clone(), best_community);
                    improved = true;
                }
            }
        }
        for (community_id, nodes) in community_nodes {
            if !nodes.is_empty() {
                let modularity = self.calculate_community_modularity(graph, &nodes);
                communities
                    .push(Community {
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
        self.louvain_community_detection(graph)
    }
    /// Label propagation algorithm
    fn label_propagation(&self, graph: &TraitGraph) -> Result<Vec<Community>> {
        let mut node_labels: HashMap<String, usize> = HashMap::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            node_labels.insert(node.id.clone(), i);
        }
        let max_iterations = 100;
        let mut rng = Random::seed(42);
        for _ in 0..max_iterations {
            let mut changed = false;
            let mut nodes = graph.nodes.clone();
            rng.shuffle(&mut nodes);
            for node in &nodes {
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
                            *label_counts.entry(neighbor_label).or_insert(0.0)
                                += edge.weight;
                        }
                    }
                }
                if let Some((&most_frequent_label, _)) = label_counts
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                {
                    let current_label = *node_labels
                        .get(&node.id)
                        .expect("get should succeed");
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
        let mut communities_map: HashMap<usize, Vec<String>> = HashMap::new();
        for (node_id, label) in node_labels {
            communities_map.entry(label).or_default().push(node_id);
        }
        let mut communities = Vec::new();
        for (label, nodes) in communities_map {
            if nodes.len() > 1 {
                let node_set: HashSet<String> = nodes.iter().cloned().collect();
                let modularity = self.calculate_community_modularity(graph, &node_set);
                communities
                    .push(Community {
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
    fn walktrap_community_detection(
        &self,
        _graph: &TraitGraph,
    ) -> Result<Vec<Community>> {
        Ok(Vec::new())
    }
    /// Girvan-Newman community detection (edge betweenness)
    fn girvan_newman_community_detection(
        &self,
        _graph: &TraitGraph,
    ) -> Result<Vec<Community>> {
        Ok(Vec::new())
    }
    /// Fast greedy community detection
    fn fast_greedy_community_detection(
        &self,
        _graph: &TraitGraph,
    ) -> Result<Vec<Community>> {
        Ok(Vec::new())
    }
    /// Find critical paths in the graph
    pub fn find_critical_paths_comprehensive(
        &self,
        graph: &TraitGraph,
    ) -> Result<Vec<GraphPath>> {
        let mut critical_paths = Vec::new();
        let centrality_measures = self.calculate_all_centrality_measures(graph)?;
        let mut high_centrality_nodes: Vec<_> = centrality_measures
            .iter()
            .filter(|(_, measures)| measures.importance_score() > 0.7)
            .map(|(id, _)| id.as_str())
            .collect();
        high_centrality_nodes
            .sort_by(|a, b| {
                let a_score = centrality_measures
                    .get(*a)
                    .map(|m| m.importance_score())
                    .unwrap_or(0.0);
                let b_score = centrality_measures
                    .get(*b)
                    .map(|m| m.importance_score())
                    .unwrap_or(0.0);
                b_score.partial_cmp(&a_score).unwrap_or(Ordering::Equal)
            });
        for i in 0..high_centrality_nodes.len().min(5) {
            for j in (i + 1)..high_centrality_nodes.len().min(5) {
                let source = high_centrality_nodes[i];
                let target = high_centrality_nodes[j];
                if let Ok(paths) = self.find_all_shortest_paths(graph, source, target) {
                    critical_paths.extend(paths);
                }
            }
        }
        critical_paths
            .sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(Ordering::Equal));
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
                            distances.insert(next_node.clone(), new_distance);
                            predecessors
                                .insert(next_node.clone(), vec![current.clone()]);
                            queue.push_back(next_node.clone());
                        }
                        Some(&existing_distance) => {
                            if new_distance < existing_distance {
                                distances.insert(next_node.clone(), new_distance);
                                predecessors
                                    .insert(next_node.clone(), vec![current.clone()]);
                            } else if (new_distance - existing_distance).abs() < 1e-10 {
                                predecessors
                                    .entry(next_node.clone())
                                    .or_default()
                                    .push(current.clone());
                            }
                        }
                    }
                }
            }
        }
        if !distances.contains_key(target) {
            return Ok(Vec::new());
        }
        let paths = self.reconstruct_all_paths(&predecessors, source, target);
        let target_distance = distances[target];
        let graph_paths = paths
            .into_iter()
            .map(|path| {
                let mut graph_path = GraphPath::new(path);
                graph_path.weight = target_distance;
                graph_path
            })
            .collect();
        Ok(graph_paths)
    }
    /// Calculate graph quality metrics
    pub fn calculate_graph_quality_metrics(
        &self,
        graph: &TraitGraph,
    ) -> Result<GraphQualityMetrics> {
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
        let node_indices: HashMap<String, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();
        for edge in &graph.edges {
            if let (Some(&from_idx), Some(&to_idx)) = (
                node_indices.get(&edge.from),
                node_indices.get(&edge.to),
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
        let node_indices: HashMap<String, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();
        let mut out_degrees = vec![0.0; n];
        for edge in &graph.edges {
            if let Some(&from_idx) = node_indices.get(&edge.from) {
                out_degrees[from_idx] += edge.weight;
            }
        }
        for edge in &graph.edges {
            if let (Some(&from_idx), Some(&to_idx)) = (
                node_indices.get(&edge.from),
                node_indices.get(&edge.to),
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
        #[derive(PartialEq)]
        struct OrderedFloat(f64);
        impl Eq for OrderedFloat {}
        impl Ord for OrderedFloat {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for OrderedFloat {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
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
                        let current_distance = distances
                            .get(next_node)
                            .copied()
                            .unwrap_or(f64::INFINITY);
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
        0.1
    }
    /// Calculate modularity for a community
    fn calculate_community_modularity(
        &self,
        graph: &TraitGraph,
        community_nodes: &HashSet<String>,
    ) -> f64 {
        if community_nodes.len() <= 1 {
            return 0.0;
        }
        let total_edges = graph.edges.len() as f64;
        if total_edges == 0.0 {
            return 0.0;
        }
        let internal_edges = graph
            .edges
            .iter()
            .filter(|edge| {
                community_nodes.contains(&edge.from)
                    && community_nodes.contains(&edge.to)
            })
            .count() as f64;
        let external_edges = graph
            .edges
            .iter()
            .filter(|edge| {
                (community_nodes.contains(&edge.from)
                    && !community_nodes.contains(&edge.to))
                    || (!community_nodes.contains(&edge.from)
                        && community_nodes.contains(&edge.to))
            })
            .count() as f64;
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
        let node_density = graph.nodes.len() as f64 / 1000.0;
        let edge_density = graph.edges.len() as f64 / (graph.nodes.len() as f64).powi(2);
        let clarity_score = 1.0
            - (node_density.min(1.0) * 0.5 + edge_density.min(1.0) * 0.5);
        clarity_score.clamp(0.0, 1.0)
    }
    /// Calculate layout quality
    fn calculate_layout_quality(&self, graph: &TraitGraph) -> f64 {
        if graph.nodes.len() < 2 {
            return 1.0;
        }
        let positioned_nodes = graph
            .nodes
            .iter()
            .filter(|node| node.position_2d.is_some())
            .count();
        if positioned_nodes == 0 {
            return 0.0;
        }
        let position_coverage = positioned_nodes as f64 / graph.nodes.len() as f64;
        let mut edge_lengths = Vec::new();
        for edge in &graph.edges {
            if let (Some(from_node), Some(to_node)) = (
                graph.nodes.iter().find(|n| n.id == edge.from),
                graph.nodes.iter().find(|n| n.id == edge.to),
            ) {
                if let (Some((x1, y1)), Some((x2, y2))) = (
                    from_node.position_2d,
                    to_node.position_2d,
                ) {
                    let length = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
                    edge_lengths.push(length);
                }
            }
        }
        let edge_length_uniformity = if edge_lengths.len() > 1 {
            let mean = edge_lengths.iter().sum::<f64>() / edge_lengths.len() as f64;
            let variance = edge_lengths.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / edge_lengths.len() as f64;
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
        let node_count = graph.nodes.len() as f64;
        let edge_count = graph.edges.len() as f64;
        let information_content = node_count + edge_count * 0.5;
        let normalized_density = information_content / (node_count * 10.0);
        normalized_density.min(1.0)
    }
    /// Calculate aesthetic appeal
    fn calculate_aesthetic_appeal(&self, graph: &TraitGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 1.0;
        }
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
        let readability = self.calculate_readability(graph);
        let navigability = self.calculate_navigability(graph);
        let interaction_design = 0.8;
        (readability + navigability + interaction_design) / 3.0
    }
    /// Calculate symmetry score (placeholder)
    fn calculate_symmetry_score(&self, _graph: &TraitGraph) -> f64 {
        0.7
    }
    /// Calculate balance score (placeholder)
    fn calculate_balance_score(&self, _graph: &TraitGraph) -> f64 {
        0.8
    }
    /// Calculate color harmony (placeholder)
    fn calculate_color_harmony(&self, _graph: &TraitGraph) -> f64 {
        0.75
    }
    /// Calculate readability (placeholder)
    fn calculate_readability(&self, _graph: &TraitGraph) -> f64 {
        0.8
    }
    /// Calculate navigability (placeholder)
    fn calculate_navigability(&self, _graph: &TraitGraph) -> f64 {
        0.85
    }
    /// Build an undirected adjacency list for the graph.
    ///
    /// Structural connectivity analyses (hub/bridge/bottleneck detection,
    /// clustering coefficients) treat every edge as an undirected
    /// connection regardless of its `directed` flag: even a directed edge
    /// `A -> B` still means removing `A` can disconnect whatever was only
    /// reachable *through* `A` from `B`'s perspective.
    fn build_undirected_adjacency(graph: &TraitGraph) -> HashMap<&str, Vec<&str>> {
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &graph.nodes {
            adjacency.entry(node.id.as_str()).or_default();
        }
        for edge in &graph.edges {
            adjacency.entry(edge.from.as_str()).or_default().push(edge.to.as_str());
            adjacency.entry(edge.to.as_str()).or_default().push(edge.from.as_str());
        }
        adjacency
    }
    /// Identify "hub" nodes: nodes whose overall centrality importance
    /// (see [`CentralityMeasures::importance_score`]) meets or exceeds
    /// `threshold` (expected range `0.0..=1.0`).
    pub fn identify_hub_nodes(
        &self,
        graph: &TraitGraph,
        threshold: f64,
    ) -> Result<Vec<String>> {
        let centrality = self.calculate_all_centrality_measures(graph)?;
        let mut hubs: Vec<String> = centrality
            .into_iter()
            .filter(|(_, measures)| measures.importance_score() >= threshold)
            .map(|(id, _)| id)
            .collect();
        hubs.sort();
        Ok(hubs)
    }
    /// Identify bridge/articulation nodes: nodes whose removal would
    /// increase the number of connected components of the graph (classic
    /// graph-theoretic "cut vertices" / articulation points).
    ///
    /// Computed via the standard Hopcroft-Tarjan DFS discovery-time /
    /// low-link algorithm over the graph's undirected adjacency (see
    /// [`Self::build_undirected_adjacency`]).
    pub fn identify_bridge_nodes(&self, graph: &TraitGraph) -> Result<Vec<String>> {
        let adjacency = Self::build_undirected_adjacency(graph);
        let mut discovery: HashMap<&str, usize> = HashMap::new();
        let mut low: HashMap<&str, usize> = HashMap::new();
        let mut articulation_points: HashSet<&str> = HashSet::new();
        let mut timer = 0usize;
        for node in &graph.nodes {
            let id = node.id.as_str();
            if discovery.contains_key(id) {
                continue;
            }
            let root_children = Self::articulation_dfs(
                id,
                None,
                &adjacency,
                &mut discovery,
                &mut low,
                &mut timer,
                &mut articulation_points,
            );
            if root_children > 1 {
                articulation_points.insert(id);
            }
        }
        let mut result: Vec<String> = articulation_points
            .into_iter()
            .map(String::from)
            .collect();
        result.sort();
        Ok(result)
    }
    /// Recursive DFS step for [`Self::identify_bridge_nodes`]. Returns the
    /// number of DFS-tree children of `node`, which the caller uses to
    /// apply the special root-node articulation-point rule (a DFS root is
    /// an articulation point iff it has more than one DFS-tree child).
    fn articulation_dfs<'a>(
        node: &'a str,
        parent: Option<&'a str>,
        adjacency: &HashMap<&'a str, Vec<&'a str>>,
        discovery: &mut HashMap<&'a str, usize>,
        low: &mut HashMap<&'a str, usize>,
        timer: &mut usize,
        articulation_points: &mut HashSet<&'a str>,
    ) -> usize {
        discovery.insert(node, *timer);
        low.insert(node, *timer);
        *timer += 1;
        let node_disc = discovery[node];
        let mut children = 0usize;
        let mut is_articulation = false;
        let mut skipped_parent_once = false;
        let neighbors = adjacency.get(node).cloned().unwrap_or_default();
        for neighbor in neighbors {
            if Some(neighbor) == parent && !skipped_parent_once {
                skipped_parent_once = true;
                continue;
            }
            if let Some(&neighbor_disc) = discovery.get(neighbor) {
                let updated = low[node].min(neighbor_disc);
                low.insert(node, updated);
            } else {
                children += 1;
                Self::articulation_dfs(
                    neighbor,
                    Some(node),
                    adjacency,
                    discovery,
                    low,
                    timer,
                    articulation_points,
                );
                let child_low = low[neighbor];
                let updated = low[node].min(child_low);
                low.insert(node, updated);
                if parent.is_some() && child_low >= node_disc {
                    is_articulation = true;
                }
            }
        }
        if is_articulation {
            articulation_points.insert(node);
        }
        children
    }
    /// Identify bottleneck edges: edges whose removal would increase the
    /// number of connected components of the graph (classic graph-theoretic
    /// "bridges", distinct from the node-level articulation points found by
    /// [`Self::identify_bridge_nodes`]).
    ///
    /// Returned as `"{from}->{to}"` identifiers matching the graph's own
    /// edge endpoints.
    pub fn identify_bottleneck_edges(&self, graph: &TraitGraph) -> Result<Vec<String>> {
        let adjacency = Self::build_undirected_adjacency(graph);
        let mut discovery: HashMap<&str, usize> = HashMap::new();
        let mut low: HashMap<&str, usize> = HashMap::new();
        let mut bridges: Vec<(&str, &str)> = Vec::new();
        let mut timer = 0usize;
        for node in &graph.nodes {
            let id = node.id.as_str();
            if discovery.contains_key(id) {
                continue;
            }
            Self::bridge_dfs(
                id,
                None,
                &adjacency,
                &mut discovery,
                &mut low,
                &mut timer,
                &mut bridges,
            );
        }
        let bridge_pairs: HashSet<(String, String)> = bridges
            .into_iter()
            .map(|(a, b)| {
                if a <= b {
                    (a.to_string(), b.to_string())
                } else {
                    (b.to_string(), a.to_string())
                }
            })
            .collect();
        let mut result: Vec<String> = graph
            .edges
            .iter()
            .filter(|edge| {
                let key = if edge.from.as_str() <= edge.to.as_str() {
                    (edge.from.clone(), edge.to.clone())
                } else {
                    (edge.to.clone(), edge.from.clone())
                };
                bridge_pairs.contains(&key)
            })
            .map(|edge| format!("{}->{}", edge.from, edge.to))
            .collect();
        result.sort();
        result.dedup();
        Ok(result)
    }
    /// Recursive DFS step for [`Self::identify_bottleneck_edges`]. An edge
    /// `(node, neighbor)` where `neighbor` is a DFS-tree child of `node` is
    /// a bridge iff `low[neighbor] > discovery[node]` (strictly greater,
    /// unlike the articulation-point condition which uses `>=`).
    fn bridge_dfs<'a>(
        node: &'a str,
        parent: Option<&'a str>,
        adjacency: &HashMap<&'a str, Vec<&'a str>>,
        discovery: &mut HashMap<&'a str, usize>,
        low: &mut HashMap<&'a str, usize>,
        timer: &mut usize,
        bridges: &mut Vec<(&'a str, &'a str)>,
    ) {
        discovery.insert(node, *timer);
        low.insert(node, *timer);
        *timer += 1;
        let node_disc = discovery[node];
        let mut skipped_parent_once = false;
        let neighbors = adjacency.get(node).cloned().unwrap_or_default();
        for neighbor in neighbors {
            if Some(neighbor) == parent && !skipped_parent_once {
                skipped_parent_once = true;
                continue;
            }
            if let Some(&neighbor_disc) = discovery.get(neighbor) {
                let updated = low[node].min(neighbor_disc);
                low.insert(node, updated);
            } else {
                Self::bridge_dfs(
                    neighbor,
                    Some(node),
                    adjacency,
                    discovery,
                    low,
                    timer,
                    bridges,
                );
                let child_low = low[neighbor];
                let updated = low[node].min(child_low);
                low.insert(node, updated);
                if child_low > node_disc {
                    bridges.push((node, neighbor));
                }
            }
        }
    }
    /// Calculate the modularity `Q` of a given community partition, using
    /// Newman's standard formula:
    ///
    /// `Q = sum_c [ (l_c / m) - (d_c / (2m))^2 ]`
    ///
    /// where for each community `c`, `l_c` is the total weight of edges
    /// internal to `c`, `d_c` is the sum of degrees (total incident edge
    /// weight) of nodes in `c`, and `m` is the total edge weight of the
    /// whole graph. Nodes not covered by `communities` do not contribute.
    pub fn calculate_modularity(
        &self,
        graph: &TraitGraph,
        communities: &[Community],
    ) -> Result<f64> {
        if graph.edges.is_empty() || communities.is_empty() {
            return Ok(0.0);
        }
        let total_weight: f64 = graph.edges.iter().map(|e| e.weight).sum();
        if total_weight <= 0.0 {
            return Ok(0.0);
        }
        let mut node_community: HashMap<&str, usize> = HashMap::new();
        for (idx, community) in communities.iter().enumerate() {
            for node_id in &community.nodes {
                node_community.insert(node_id.as_str(), idx);
            }
        }
        let mut degree: HashMap<&str, f64> = HashMap::new();
        for edge in &graph.edges {
            *degree.entry(edge.from.as_str()).or_insert(0.0) += edge.weight;
            *degree.entry(edge.to.as_str()).or_insert(0.0) += edge.weight;
        }
        let mut internal_weight = vec![0.0; communities.len()];
        for edge in &graph.edges {
            if let (Some(&from_c), Some(&to_c)) = (
                node_community.get(edge.from.as_str()),
                node_community.get(edge.to.as_str()),
            ) {
                if from_c == to_c {
                    internal_weight[from_c] += edge.weight;
                }
            }
        }
        let two_m = 2.0 * total_weight;
        let q: f64 = communities
            .iter()
            .enumerate()
            .map(|(idx, community)| {
                let l_c = internal_weight[idx];
                let d_c: f64 = community
                    .nodes
                    .iter()
                    .filter_map(|id| degree.get(id.as_str()))
                    .sum();
                (l_c / total_weight) - (d_c / two_m).powi(2)
            })
            .sum();
        Ok(q)
    }
    /// Estimate the small-world coefficient (sigma) of the graph: the ratio
    /// of its average clustering coefficient to that of an equivalent
    /// random (Erdos-Renyi) graph, divided by the ratio of its average
    /// shortest-path length to that of an equivalent random graph.
    ///
    /// `sigma > 1` indicates small-world structure (high local clustering
    /// combined with short average path length), which is common in
    /// well-factored trait hierarchies (tight local clusters of related
    /// traits, bridged by a few widely-used foundational traits).
    pub fn calculate_small_world_coefficient(&self, graph: &TraitGraph) -> Result<f64> {
        let n = graph.nodes.len();
        if n < 3 {
            return Ok(0.0);
        }
        let clustering = self.calculate_average_clustering_coefficient(graph);
        let mut total_distance = 0.0;
        let mut pair_count = 0usize;
        for node in &graph.nodes {
            let distances = self.calculate_shortest_path_distances(graph, &node.id)?;
            for (target, distance) in &distances {
                if target != &node.id {
                    total_distance += distance;
                    pair_count += 1;
                }
            }
        }
        if pair_count == 0 {
            return Ok(0.0);
        }
        let avg_path_length = total_distance / pair_count as f64;
        let avg_degree = (2.0 * graph.edges.len() as f64 / n as f64).max(1e-9);
        let random_clustering = (avg_degree / n as f64).max(1e-9);
        let random_path_length = if avg_degree > 1.0 {
            (n as f64).ln() / avg_degree.ln()
        } else {
            n as f64
        };
        if random_clustering <= 0.0 || random_path_length <= 0.0
            || avg_path_length <= 0.0
        {
            return Ok(0.0);
        }
        let sigma = (clustering / random_clustering)
            / (avg_path_length / random_path_length);
        Ok(sigma)
    }
    /// Average (undirected) clustering coefficient across all nodes with at
    /// least two neighbors: for each such node, the fraction of pairs of
    /// its neighbors that are themselves directly connected.
    fn calculate_average_clustering_coefficient(&self, graph: &TraitGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 0.0;
        }
        let mut adjacency: HashMap<&str, HashSet<&str>> = HashMap::new();
        for node in &graph.nodes {
            adjacency.entry(node.id.as_str()).or_default();
        }
        for edge in &graph.edges {
            adjacency.entry(edge.from.as_str()).or_default().insert(edge.to.as_str());
            adjacency.entry(edge.to.as_str()).or_default().insert(edge.from.as_str());
        }
        let mut total = 0.0;
        let mut counted = 0usize;
        for node in &graph.nodes {
            let neighbors = match adjacency.get(node.id.as_str()) {
                Some(set) if set.len() >= 2 => set,
                _ => continue,
            };
            let neighbor_vec: Vec<&str> = neighbors.iter().copied().collect();
            let mut links = 0usize;
            for i in 0..neighbor_vec.len() {
                for j in (i + 1)..neighbor_vec.len() {
                    if adjacency
                        .get(neighbor_vec[i])
                        .map(|set| set.contains(neighbor_vec[j]))
                        .unwrap_or(false)
                    {
                        links += 1;
                    }
                }
            }
            let possible = neighbor_vec.len() * (neighbor_vec.len() - 1) / 2;
            if possible > 0 {
                total += links as f64 / possible as f64;
                counted += 1;
            }
        }
        if counted == 0 { 0.0 } else { total / counted as f64 }
    }
    /// Enable or disable SIMD optimization
    pub fn set_simd_enabled(&mut self, enabled: bool) {
        self.simd_enabled = enabled;
    }
    /// Enable or disable parallel processing
    pub fn set_parallel_enabled(&mut self, enabled: bool) {
        self.parallel_enabled = enabled;
    }
    /// Get performance statistics: average duration per recorded operation,
    /// across all timing categories (centrality, community detection, and
    /// path-finding).
    pub fn get_performance_stats(&self) -> Option<HashMap<String, std::time::Duration>> {
        if let Ok(tracker) = self.performance_tracker.lock() {
            let mut stats = HashMap::new();
            for operation in tracker.timed_operations() {
                if let Some(avg_time) = tracker.get_average_timing(operation) {
                    stats.insert(operation.clone(), avg_time);
                }
            }
            Some(stats)
        } else {
            None
        }
    }
    /// Get average recorded memory usage (in bytes) per operation.
    pub fn get_memory_stats(&self) -> Option<HashMap<String, u64>> {
        if let Ok(tracker) = self.performance_tracker.lock() {
            let mut stats = HashMap::new();
            for operation in tracker.memory_usage.keys() {
                if let Some(avg_bytes) = tracker.get_average_memory_usage(operation) {
                    stats.insert(operation.clone(), avg_bytes);
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

#[cfg(test)]
mod tests;
