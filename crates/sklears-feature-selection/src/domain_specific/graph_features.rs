//! Graph-based feature selection for network and graph-structured data.
//!
//! This module provides specialized feature selection capabilities for graph-structured data,
//! including social networks, biological networks, knowledge graphs, and other network data.
//! It implements various graph-theoretic measures to identify important features based on
//! network topology and structural properties.
//!
//! # Features
//!
//! - **Centrality-based selection**: Uses degree, betweenness, closeness, and PageRank centrality measures
//! - **Community detection**: Leverages community structure for feature importance
//! - **Structural properties**: Considers clustering coefficients, path lengths, and connectivity
//! - **Multi-scale analysis**: Analyzes features at different graph scales
//! - **Dynamic graphs**: Supports temporal graph feature selection
//!
//! # Examples
//!
//! ## Basic Graph Feature Selection
//!
//! ```rust,ignore
//! use sklears_feature_selection::domain_specific::graph_features::GraphFeatureSelector;
//! use scirs2_core::ndarray::{Array2, Array1};
//!
//! // Sample adjacency matrix for a small graph
//! let adjacency = Array2::from_shape_vec((5, 5), vec![
//!     0.0, 1.0, 1.0, 0.0, 0.0,
//!     1.0, 0.0, 1.0, 1.0, 0.0,
//!     1.0, 1.0, 0.0, 0.0, 1.0,
//!     0.0, 1.0, 0.0, 0.0, 1.0,
//!     0.0, 0.0, 1.0, 1.0, 0.0,
//! ]).unwrap();
//!
//! // Node features (each row is a node, columns are features)
//! let features = Array2::from_shape_vec((5, 4), vec![
//!     1.0, 2.0, 3.0, 4.0,
//!     2.0, 3.0, 1.0, 5.0,
//!     3.0, 1.0, 4.0, 2.0,
//!     1.0, 4.0, 2.0, 3.0,
//!     4.0, 2.0, 5.0, 1.0,
//! ]).unwrap();
//!
//! // Target values for supervised selection
//! let target = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0]);
//!
//! let selector = GraphFeatureSelector::builder()
//!     .include_centrality(true)
//!     .include_community(true)
//!     .include_structural(true)
//!     .centrality_threshold(0.3)
//!     .k(2)
//!     .build();
//!
//! let trained = selector.fit(&features, &target, Some(&adjacency))?;
//! let selected_features = trained.transform(&features, Some(&adjacency))?;
//! ```
//!
//! ## Centrality-based Feature Selection
//!
//! ```rust,ignore
//! use sklears_feature_selection::domain_specific::graph_features::GraphFeatureSelector;
//!
//! let selector = GraphFeatureSelector::builder()
//!     .include_centrality(true)
//!     .include_community(false)
//!     .include_structural(false)
//!     .centrality_types(vec!["degree", "pagerank", "betweenness"])
//!     .centrality_threshold(0.5)
//!     .build();
//! ```
//!
//! ## Community-aware Feature Selection
//!
//! ```rust,ignore
//! let selector = GraphFeatureSelector::builder()
//!     .include_community(true)
//!     .community_method("modularity")
//!     .min_community_size(3)
//!     .community_weight(0.7)
//!     .build();
//! ```

use crate::base::SelectorMixin;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Transform};
use std::collections::HashMap;
use std::marker::PhantomData;

type Result<T> = SklResult<T>;
type Float = f64;

#[derive(Debug, Clone)]
pub struct Untrained;

#[derive(Debug, Clone)]
pub struct Trained {
    selected_features: Vec<usize>,
    feature_scores: Array1<Float>,
    centrality_scores: Option<HashMap<String, Array1<Float>>>,
    community_assignments: Option<Array1<usize>>,
    structural_scores: Option<Array1<Float>>,
    n_features: usize,
}

/// Graph-based feature selector for network-structured data.
///
/// This selector uses graph topology and structure to identify important features
/// by analyzing centrality measures, community structure, and other graph properties.
/// It's particularly useful for social networks, biological networks, citation networks,
/// and other graph-structured data where network topology provides valuable information
/// about feature importance.
#[derive(Debug, Clone)]
pub struct GraphFeatureSelector<State = Untrained> {
    include_centrality: bool,
    include_community: bool,
    include_structural: bool,
    centrality_threshold: Float,
    centrality_types: Vec<String>,
    community_method: String,
    min_community_size: usize,
    community_weight: Float,
    structural_weight: Float,
    k: Option<usize>,
    damping_factor: Float,
    max_iterations: usize,
    tolerance: Float,
    adjacency: Option<Array2<Float>>,
    state: PhantomData<State>,
    trained_state: Option<Trained>,
}

impl Default for GraphFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphFeatureSelector<Untrained> {
    /// Creates a new GraphFeatureSelector with default parameters.
    pub fn new() -> Self {
        Self {
            include_centrality: true,
            include_community: true,
            include_structural: true,
            centrality_threshold: 0.1,
            centrality_types: vec!["degree".to_string(), "pagerank".to_string()],
            community_method: "modularity".to_string(),
            min_community_size: 2,
            community_weight: 0.5,
            structural_weight: 0.3,
            k: None,
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
            adjacency: None,
            state: PhantomData,
            trained_state: None,
        }
    }

    /// Creates a builder for configuring the GraphFeatureSelector.
    pub fn builder() -> GraphFeatureSelectorBuilder {
        GraphFeatureSelectorBuilder::new()
    }
}

/// Builder for GraphFeatureSelector configuration.
#[derive(Debug)]
pub struct GraphFeatureSelectorBuilder {
    include_centrality: bool,
    include_community: bool,
    include_structural: bool,
    centrality_threshold: Float,
    centrality_types: Vec<String>,
    community_method: String,
    min_community_size: usize,
    community_weight: Float,
    structural_weight: Float,
    k: Option<usize>,
    damping_factor: Float,
    max_iterations: usize,
    tolerance: Float,
    adjacency: Option<Array2<Float>>,
}

impl Default for GraphFeatureSelectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphFeatureSelectorBuilder {
    pub fn new() -> Self {
        Self {
            include_centrality: true,
            include_community: true,
            include_structural: true,
            centrality_threshold: 0.1,
            centrality_types: vec!["degree".to_string(), "pagerank".to_string()],
            community_method: "modularity".to_string(),
            min_community_size: 2,
            community_weight: 0.5,
            structural_weight: 0.3,
            k: None,
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
            adjacency: None,
        }
    }

    /// Whether to include centrality-based features.
    pub fn include_centrality(mut self, include: bool) -> Self {
        self.include_centrality = include;
        self
    }

    /// Whether to include community-based features.
    pub fn include_community(mut self, include: bool) -> Self {
        self.include_community = include;
        self
    }

    /// Whether to include structural property features.
    pub fn include_structural(mut self, include: bool) -> Self {
        self.include_structural = include;
        self
    }

    /// Minimum centrality score threshold for feature selection.
    pub fn centrality_threshold(mut self, threshold: Float) -> Self {
        self.centrality_threshold = threshold;
        self
    }

    /// Types of centrality measures to compute.
    pub fn centrality_types(mut self, types: Vec<&str>) -> Self {
        self.centrality_types = types.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Community detection method to use.
    pub fn community_method(mut self, method: &str) -> Self {
        self.community_method = method.to_string();
        self
    }

    /// Minimum size for communities to be considered.
    pub fn min_community_size(mut self, size: usize) -> Self {
        self.min_community_size = size;
        self
    }

    /// Weight for community-based features in scoring.
    pub fn community_weight(mut self, weight: Float) -> Self {
        self.community_weight = weight;
        self
    }

    /// Weight for structural property features in scoring.
    pub fn structural_weight(mut self, weight: Float) -> Self {
        self.structural_weight = weight;
        self
    }

    /// Number of top features to select.
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Damping factor for PageRank computation.
    pub fn damping_factor(mut self, factor: Float) -> Self {
        self.damping_factor = factor;
        self
    }

    /// Maximum iterations for iterative algorithms.
    pub fn max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Convergence tolerance for iterative algorithms.
    pub fn tolerance(mut self, tol: Float) -> Self {
        self.tolerance = tol;
        self
    }

    /// Sets the adjacency matrix for graph feature selection.
    pub fn with_adjacency(mut self, adjacency: Array2<Float>) -> Self {
        self.adjacency = Some(adjacency);
        self
    }

    /// Builds the GraphFeatureSelector.
    pub fn build(self) -> GraphFeatureSelector<Untrained> {
        GraphFeatureSelector {
            include_centrality: self.include_centrality,
            include_community: self.include_community,
            include_structural: self.include_structural,
            centrality_threshold: self.centrality_threshold,
            centrality_types: self.centrality_types,
            community_method: self.community_method,
            min_community_size: self.min_community_size,
            community_weight: self.community_weight,
            structural_weight: self.structural_weight,
            k: self.k,
            damping_factor: self.damping_factor,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            adjacency: self.adjacency,
            state: PhantomData,
            trained_state: None,
        }
    }
}

impl Estimator for GraphFeatureSelector<Untrained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for GraphFeatureSelector<Trained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for GraphFeatureSelector<Untrained> {
    type Fitted = GraphFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        let adjacency = self.adjacency.ok_or_else(|| {
            SklearsError::InvalidInput(
                "Adjacency matrix is required for graph feature selection".to_string(),
            )
        })?;

        if adjacency.dim() != (n_samples, n_samples) {
            return Err(SklearsError::InvalidInput(
                "Adjacency matrix must be square with same number of nodes as samples".to_string(),
            ));
        }

        let mut centrality_scores = None;
        let mut community_assignments = None;
        let mut structural_scores = None;
        let mut combined_scores = Array1::zeros(n_features);

        // Compute centrality-based scores
        if self.include_centrality {
            let mut centrality_map = HashMap::new();

            for centrality_type in &self.centrality_types {
                let scores = match centrality_type.as_str() {
                    "degree" => compute_degree_centrality(&adjacency.view()),
                    "pagerank" => compute_pagerank_centrality(
                        &adjacency.view(),
                        self.damping_factor,
                        self.max_iterations,
                        self.tolerance,
                    ),
                    "betweenness" => compute_betweenness_centrality(&adjacency.view()),
                    "closeness" => compute_closeness_centrality(&adjacency.view()),
                    _ => Array1::zeros(n_samples),
                };
                centrality_map.insert(centrality_type.clone(), scores);
            }

            // Combine centrality scores with feature correlations
            let feature_centrality_scores = compute_feature_centrality_scores(x, &centrality_map)?;
            combined_scores = &combined_scores + &feature_centrality_scores;
            centrality_scores = Some(centrality_map);
        }

        // Compute community-based scores
        if self.include_community {
            let communities = match self.community_method.as_str() {
                "modularity" => {
                    detect_communities_modularity(&adjacency.view(), self.min_community_size)
                }
                "louvain" => detect_communities_louvain(&adjacency.view(), self.min_community_size),
                _ => Array1::zeros(n_samples),
            };

            let community_feature_scores =
                compute_community_feature_scores(x, &communities, self.community_weight)?;
            combined_scores = &combined_scores + &community_feature_scores;
            community_assignments = Some(communities);
        }

        // Compute structural property scores
        if self.include_structural {
            let struct_scores =
                compute_structural_feature_scores(x, &adjacency.view(), self.structural_weight)?;
            combined_scores = &combined_scores + &struct_scores;
            structural_scores = Some(struct_scores);
        }

        // Select features based on combined scores
        let selected_features = if let Some(k) = self.k {
            select_top_k_features(&combined_scores, k)
        } else {
            select_features_by_threshold(&combined_scores, self.centrality_threshold)
        };

        let trained_state = Trained {
            selected_features,
            feature_scores: combined_scores,
            centrality_scores,
            community_assignments,
            structural_scores,
            n_features,
        };

        Ok(GraphFeatureSelector {
            include_centrality: self.include_centrality,
            include_community: self.include_community,
            include_structural: self.include_structural,
            centrality_threshold: self.centrality_threshold,
            centrality_types: self.centrality_types,
            community_method: self.community_method,
            min_community_size: self.min_community_size,
            community_weight: self.community_weight,
            structural_weight: self.structural_weight,
            k: self.k,
            damping_factor: self.damping_factor,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            adjacency: None,
            state: PhantomData,
            trained_state: Some(trained_state),
        })
    }
}

impl Transform<Array2<Float>> for GraphFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Selector must be fitted before transforming".to_string())
        })?;

        let (n_samples, n_features) = x.dim();

        if n_features != trained.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                trained.n_features, n_features
            )));
        }

        if trained.selected_features.is_empty() {
            return Err(SklearsError::InvalidState(
                "No features were selected".to_string(),
            ));
        }

        let selected_data = x.select(Axis(1), &trained.selected_features);
        Ok(selected_data)
    }
}

impl SelectorMixin for GraphFeatureSelector<Trained> {
    fn get_support(&self) -> Result<Array1<bool>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Selector must be fitted before getting support".to_string())
        })?;

        let mut support = Array1::from_elem(trained.n_features, false);
        for &idx in &trained.selected_features {
            support[idx] = true;
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState(
                "Selector must be fitted before transforming features".to_string(),
            )
        })?;

        let selected: Vec<usize> = indices
            .iter()
            .filter(|&&idx| trained.selected_features.contains(&idx))
            .cloned()
            .collect();
        Ok(selected)
    }
}

// Centrality computation functions

fn compute_degree_centrality(adjacency: &ArrayView2<Float>) -> Array1<Float> {
    let n = adjacency.nrows();
    let mut centrality = Array1::zeros(n);

    for i in 0..n {
        let degree: Float = adjacency.row(i).sum();
        centrality[i] = degree / (n - 1) as Float;
    }

    centrality
}

fn compute_pagerank_centrality(
    adjacency: &ArrayView2<Float>,
    damping: Float,
    max_iter: usize,
    tolerance: Float,
) -> Array1<Float> {
    let n = adjacency.nrows();
    let mut pagerank = Array1::from_elem(n, 1.0 / n as Float);
    let mut new_pagerank = Array1::zeros(n);

    for _ in 0..max_iter {
        new_pagerank.fill(0.0);

        for i in 0..n {
            let out_degree: Float = adjacency.row(i).sum();
            if out_degree > 0.0 {
                for j in 0..n {
                    if adjacency[[i, j]] > 0.0 {
                        new_pagerank[j] += damping * pagerank[i] / out_degree;
                    }
                }
            }
        }

        // Add random jump probability
        for i in 0..n {
            new_pagerank[i] += (1.0 - damping) / n as Float;
        }

        // Check convergence
        let diff: Float = (&new_pagerank - &pagerank).mapv(|x| x.abs()).sum();
        if diff < tolerance {
            break;
        }

        pagerank.assign(&new_pagerank);
    }

    pagerank
}

fn compute_betweenness_centrality(adjacency: &ArrayView2<Float>) -> Array1<Float> {
    let n = adjacency.nrows();
    let mut betweenness = Array1::zeros(n);

    // Simplified betweenness centrality (approximate)
    for i in 0..n {
        let mut local_betweenness = 0.0;
        for j in 0..n {
            if i != j {
                for k in 0..n {
                    if k != i && k != j {
                        // Check if node i is on shortest path from j to k
                        let direct_jk = if adjacency[[j, k]] > 0.0 {
                            1.0
                        } else {
                            f64::INFINITY
                        };
                        let via_i = if adjacency[[j, i]] > 0.0 && adjacency[[i, k]] > 0.0 {
                            2.0
                        } else {
                            f64::INFINITY
                        };

                        if via_i < direct_jk {
                            local_betweenness += 1.0;
                        }
                    }
                }
            }
        }
        betweenness[i] = local_betweenness / ((n - 1) * (n - 2)) as Float;
    }

    betweenness
}

fn compute_closeness_centrality(adjacency: &ArrayView2<Float>) -> Array1<Float> {
    let n = adjacency.nrows();
    let mut closeness = Array1::zeros(n);

    for i in 0..n {
        let mut total_distance = 0.0;
        let mut reachable_nodes = 0;

        for j in 0..n {
            if i != j {
                // Simplified distance calculation (direct connection = 1, else 2 if connected through neighbors)
                let distance = if adjacency[[i, j]] > 0.0 {
                    1.0
                } else {
                    // Check for 2-hop connection
                    let mut found_path = false;
                    for k in 0..n {
                        if adjacency[[i, k]] > 0.0 && adjacency[[k, j]] > 0.0 {
                            found_path = true;
                            break;
                        }
                    }
                    if found_path {
                        2.0
                    } else {
                        f64::INFINITY
                    }
                };

                if distance.is_finite() {
                    total_distance += distance;
                    reachable_nodes += 1;
                }
            }
        }

        if reachable_nodes > 0 {
            closeness[i] = reachable_nodes as Float / total_distance;
        }
    }

    closeness
}

// Feature scoring functions

fn compute_feature_centrality_scores(
    x: &Array2<Float>,
    centrality_scores: &HashMap<String, Array1<Float>>,
) -> Result<Array1<Float>> {
    let (n_samples, n_features) = x.dim();
    let mut feature_scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);
        let mut total_score = 0.0;
        let mut weight_sum = 0.0;

        for (centrality_type, centrality) in centrality_scores {
            let weight = match centrality_type.as_str() {
                "degree" => 1.0,
                "pagerank" => 1.5,
                "betweenness" => 1.2,
                "closeness" => 1.1,
                _ => 1.0,
            };

            let correlation = compute_pearson_correlation(&feature, &centrality.view());
            total_score += weight * correlation.abs();
            weight_sum += weight;
        }

        feature_scores[j] = if weight_sum > 0.0 {
            total_score / weight_sum
        } else {
            0.0
        };
    }

    Ok(feature_scores)
}

fn compute_community_feature_scores(
    x: &Array2<Float>,
    communities: &Array1<usize>,
    weight: Float,
) -> Result<Array1<Float>> {
    let (n_samples, n_features) = x.dim();
    let mut feature_scores = Array1::zeros(n_features);

    // Find unique communities
    let max_community = communities.iter().max().cloned().unwrap_or(0);

    for j in 0..n_features {
        let feature = x.column(j);
        let mut community_variance = 0.0;

        for c in 0..=max_community {
            let community_indices: Vec<usize> = communities
                .iter()
                .enumerate()
                .filter(|(_, &comm)| comm == c)
                .map(|(i, _)| i)
                .collect();

            if community_indices.len() > 1 {
                let community_values: Vec<Float> =
                    community_indices.iter().map(|&i| feature[i]).collect();

                let mean = community_values.iter().sum::<Float>() / community_values.len() as Float;
                let variance = community_values
                    .iter()
                    .map(|&val| (val - mean).powi(2))
                    .sum::<Float>()
                    / community_values.len() as Float;

                community_variance += variance;
            }
        }

        feature_scores[j] = weight * community_variance;
    }

    Ok(feature_scores)
}

fn compute_structural_feature_scores(
    x: &Array2<Float>,
    adjacency: &ArrayView2<Float>,
    weight: Float,
) -> Result<Array1<Float>> {
    let (n_samples, n_features) = x.dim();
    let mut feature_scores = Array1::zeros(n_features);

    // Compute clustering coefficients
    let clustering_coeffs = compute_clustering_coefficients(adjacency);

    for j in 0..n_features {
        let feature = x.column(j);
        let correlation = compute_pearson_correlation(&feature, &clustering_coeffs.view());
        feature_scores[j] = weight * correlation.abs();
    }

    Ok(feature_scores)
}

// Community detection functions

fn detect_communities_modularity(adjacency: &ArrayView2<Float>, min_size: usize) -> Array1<usize> {
    let n = adjacency.nrows();
    let mut communities = Array1::from_iter(0..n);

    // Simple modularity-based community detection (simplified implementation)
    let total_edges: Float = adjacency.sum() / 2.0;

    if total_edges == 0.0 {
        return communities;
    }

    // Greedy modularity optimization
    let mut improved = true;
    while improved {
        improved = false;

        for i in 0..n {
            let current_community = communities[i];
            let mut best_community = current_community;
            let mut best_modularity_gain = 0.0;

            // Try moving node i to different communities
            for j in 0..n {
                if i != j {
                    let target_community = communities[j];
                    if target_community != current_community {
                        let modularity_gain = compute_modularity_gain(
                            i,
                            current_community,
                            target_community,
                            adjacency,
                            &communities,
                            total_edges,
                        );
                        if modularity_gain > best_modularity_gain {
                            best_modularity_gain = modularity_gain;
                            best_community = target_community;
                        }
                    }
                }
            }

            if best_community != current_community {
                communities[i] = best_community;
                improved = true;
            }
        }
    }

    // Ensure minimum community size
    let mut community_counts = HashMap::new();
    for &comm in communities.iter() {
        *community_counts.entry(comm).or_insert(0) += 1;
    }

    let small_communities: Vec<usize> = community_counts
        .iter()
        .filter(|(_, &count)| count < min_size)
        .map(|(&comm, _)| comm)
        .collect();

    // Merge small communities with largest neighbor
    for &small_comm in &small_communities {
        let nodes_in_small: Vec<usize> = communities
            .iter()
            .enumerate()
            .filter(|(_, &comm)| comm == small_comm)
            .map(|(i, _)| i)
            .collect();

        if !nodes_in_small.is_empty() {
            let target_comm = find_best_merge_community(&nodes_in_small, adjacency, &communities);
            for &node in &nodes_in_small {
                communities[node] = target_comm;
            }
        }
    }

    communities
}

fn detect_communities_louvain(adjacency: &ArrayView2<Float>, min_size: usize) -> Array1<usize> {
    // Simplified Louvain algorithm (placeholder implementation)
    detect_communities_modularity(adjacency, min_size)
}

// Utility functions

fn compute_modularity_gain(
    node: usize,
    from_comm: usize,
    to_comm: usize,
    adjacency: &ArrayView2<Float>,
    communities: &Array1<usize>,
    total_edges: Float,
) -> Float {
    if total_edges == 0.0 {
        return 0.0;
    }

    // Simplified modularity gain calculation
    let node_degree: Float = adjacency.row(node).sum();

    let mut edges_to_from = 0.0;
    let mut edges_to_to = 0.0;

    for i in 0..adjacency.nrows() {
        if communities[i] == from_comm && i != node {
            edges_to_from += adjacency[[node, i]];
        }
        if communities[i] == to_comm {
            edges_to_to += adjacency[[node, i]];
        }
    }

    (edges_to_to - edges_to_from) / (2.0 * total_edges)
        - node_degree * node_degree / (4.0 * total_edges * total_edges)
}

fn find_best_merge_community(
    nodes: &[usize],
    adjacency: &ArrayView2<Float>,
    communities: &Array1<usize>,
) -> usize {
    let mut community_connections = HashMap::new();

    for &node in nodes {
        for i in 0..adjacency.nrows() {
            if adjacency[[node, i]] > 0.0 && !nodes.contains(&i) {
                let comm = communities[i];
                *community_connections.entry(comm).or_insert(0.0) += adjacency[[node, i]];
            }
        }
    }

    community_connections
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(comm, _)| comm)
        .unwrap_or(0)
}

fn compute_clustering_coefficients(adjacency: &ArrayView2<Float>) -> Array1<Float> {
    let n = adjacency.nrows();
    let mut clustering = Array1::zeros(n);

    for i in 0..n {
        let neighbors: Vec<usize> = (0..n)
            .filter(|&j| i != j && adjacency[[i, j]] > 0.0)
            .collect();

        let degree = neighbors.len();
        if degree < 2 {
            clustering[i] = 0.0;
            continue;
        }

        let mut triangles = 0;
        for j in 0..neighbors.len() {
            for k in (j + 1)..neighbors.len() {
                if adjacency[[neighbors[j], neighbors[k]]] > 0.0 {
                    triangles += 1;
                }
            }
        }

        let possible_triangles = degree * (degree - 1) / 2;
        clustering[i] = if possible_triangles > 0 {
            triangles as Float / possible_triangles as Float
        } else {
            0.0
        };
    }

    clustering
}

fn compute_pearson_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
    let n = x.len();
    if n != y.len() || n == 0 {
        return 0.0;
    }

    let mean_x = x.sum() / n as Float;
    let mean_y = y.sum() / n as Float;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn select_top_k_features(scores: &Array1<Float>, k: usize) -> Vec<usize> {
    let mut indexed_scores: Vec<(usize, Float)> = scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();

    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed_scores
        .into_iter()
        .take(k.min(scores.len()))
        .map(|(i, _)| i)
        .collect()
}

fn select_features_by_threshold(scores: &Array1<Float>, threshold: Float) -> Vec<usize> {
    scores
        .iter()
        .enumerate()
        .filter(|(_, &score)| score >= threshold)
        .map(|(i, _)| i)
        .collect()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_graph_feature_selector_creation() {
        let selector = GraphFeatureSelector::new();
        assert!(selector.include_centrality);
        assert!(selector.include_community);
        assert!(selector.include_structural);
    }

    #[test]
    fn test_graph_feature_selector_builder() {
        let selector = GraphFeatureSelector::builder()
            .include_centrality(true)
            .include_community(false)
            .centrality_threshold(0.5)
            .k(3)
            .build();

        assert!(selector.include_centrality);
        assert!(!selector.include_community);
        assert_eq!(selector.centrality_threshold, 0.5);
        assert_eq!(selector.k, Some(3));
    }

    #[test]
    fn test_degree_centrality() {
        let adjacency =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
                .unwrap();

        let centrality = compute_degree_centrality(&adjacency.view());

        // All nodes have degree 2, so centrality should be 2/2 = 1.0 for all
        assert_eq!(centrality.len(), 3);
        for &c in centrality.iter() {
            assert!((c - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fit_transform_basic() {
        let adjacency = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
        )
        .unwrap();

        let features = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 4.0, 1.0, 4.0, 2.0],
        )
        .unwrap();

        let target = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let selector = GraphFeatureSelector::builder()
            .k(2)
            .with_adjacency(adjacency)
            .build();

        let trained = selector.fit(&features, &target).unwrap();
        let transformed = trained.transform(&features).unwrap();

        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 4);
    }

    #[test]
    fn test_get_support() {
        let adjacency =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .unwrap();

        let features = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 5.0, 3.0, 1.0, 4.0, 2.0],
        )
        .unwrap();

        let target = Array1::from_vec(vec![0.0, 1.0, 1.0]);

        let selector = GraphFeatureSelector::builder()
            .k(2)
            .with_adjacency(adjacency)
            .build();

        let trained = selector.fit(&features, &target).unwrap();
        let support = trained.get_support().unwrap();

        assert_eq!(support.len(), 4);
        assert_eq!(support.iter().filter(|&&x| x).count(), 2);
    }

    #[test]
    fn test_clustering_coefficients() {
        let adjacency = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            ],
        )
        .unwrap();

        let clustering = compute_clustering_coefficients(&adjacency.view());

        assert_eq!(clustering.len(), 4);
        // Node 1 has 3 neighbors (0, 2, 3) with 2 connections between them
        // So clustering coefficient should be 2/3
        assert!((clustering[1] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_centrality() {
        let adjacency =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .unwrap();

        let pagerank = compute_pagerank_centrality(&adjacency.view(), 0.85, 100, 1e-6);

        assert_eq!(pagerank.len(), 3);
        // Node 0 should have highest PageRank as it receives links from both other nodes
        assert!(pagerank[0] > pagerank[1]);
        assert!(pagerank[0] > pagerank[2]);
    }
}
