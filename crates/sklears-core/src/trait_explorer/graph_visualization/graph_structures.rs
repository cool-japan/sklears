//! Core graph data structures for trait visualization
//!
//! This module provides the fundamental data structures used throughout
//! the graph visualization system, including nodes, edges, and the main
//! graph container with associated metadata and statistics.

use super::graph_config::{TraitNodeType, EdgeType, StabilityLevel, CommunityDetection};
use crate::api_reference_generator::{AssociatedType, TraitInfo};
use crate::error::{Result, SklearsError};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Node in the trait graph with enhanced properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraitGraphNode {
    /// Unique identifier for the node
    pub id: String,
    /// Display label for the node
    pub label: String,
    /// Type of the node (trait, implementation, etc.)
    pub node_type: TraitNodeType,
    /// Position in 2D space (x, y)
    pub position_2d: Option<(f64, f64)>,
    /// Position in 3D space (x, y, z)
    pub position_3d: Option<(f64, f64, f64)>,
    /// Visual size of the node
    pub size: f64,
    /// Visual color of the node
    pub color: Option<String>,
    /// Shape of the node for rendering
    pub shape: Option<String>,
    /// Visibility flag
    pub visible: bool,
    /// Additional metadata
    pub metadata: NodeMetadata,
}

impl TraitGraphNode {
    /// Create a new trait node
    pub fn new_trait(id: String, label: String) -> Self {
        Self {
            id,
            label,
            node_type: TraitNodeType::Trait,
            position_2d: None,
            position_3d: None,
            size: 1.0,
            color: Some(TraitNodeType::Trait.default_color().to_string()),
            shape: Some(TraitNodeType::Trait.default_shape().to_string()),
            visible: true,
            metadata: NodeMetadata::default(),
        }
    }

    /// Create a new implementation node
    pub fn new_implementation(id: String, label: String, trait_name: String) -> Self {
        Self {
            id,
            label,
            node_type: TraitNodeType::Implementation,
            position_2d: None,
            position_3d: None,
            size: 0.8,
            color: Some(TraitNodeType::Implementation.default_color().to_string()),
            shape: Some(TraitNodeType::Implementation.default_shape().to_string()),
            visible: true,
            metadata: NodeMetadata {
                trait_name: Some(trait_name),
                ..Default::default()
            },
        }
    }

    /// Create a new associated type node
    pub fn new_associated_type(id: String, label: String, trait_name: String) -> Self {
        Self {
            id,
            label,
            node_type: TraitNodeType::AssociatedType,
            position_2d: None,
            position_3d: None,
            size: 0.6,
            color: Some(TraitNodeType::AssociatedType.default_color().to_string()),
            shape: Some(TraitNodeType::AssociatedType.default_shape().to_string()),
            visible: true,
            metadata: NodeMetadata {
                trait_name: Some(trait_name),
                ..Default::default()
            },
        }
    }

    /// Set the position in 2D space
    pub fn with_position_2d(mut self, x: f64, y: f64) -> Self {
        self.position_2d = Some((x, y));
        self
    }

    /// Set the position in 3D space
    pub fn with_position_3d(mut self, x: f64, y: f64, z: f64) -> Self {
        self.position_3d = Some((x, y, z));
        self
    }

    /// Set the visual size
    pub fn with_size(mut self, size: f64) -> Self {
        self.size = size;
        self
    }

    /// Set the visual color
    pub fn with_color(mut self, color: String) -> Self {
        self.color = Some(color);
        self
    }

    /// Set the stability level
    pub fn with_stability(mut self, stability: StabilityLevel) -> Self {
        self.metadata.stability = stability;
        self
    }

    /// Set the complexity score
    pub fn with_complexity(mut self, complexity: f64) -> Self {
        self.metadata.complexity = complexity;
        self
    }

    /// Check if this node is a trait
    pub fn is_trait(&self) -> bool {
        self.node_type == TraitNodeType::Trait
    }

    /// Check if this node is an implementation
    pub fn is_implementation(&self) -> bool {
        self.node_type == TraitNodeType::Implementation
    }

    /// Check if this node is an associated type
    pub fn is_associated_type(&self) -> bool {
        self.node_type == TraitNodeType::AssociatedType
    }

    /// Get the display title including type information
    pub fn display_title(&self) -> String {
        format!("{} ({})", self.label, self.node_type.display_name())
    }

    /// Calculate the visual importance score
    pub fn importance_score(&self) -> f64 {
        let base_score = match self.node_type {
            TraitNodeType::Trait => 1.0,
            TraitNodeType::Implementation => 0.8,
            TraitNodeType::AssociatedType => 0.6,
            TraitNodeType::Method => 0.4,
            TraitNodeType::Constant => 0.3,
            _ => 0.2,
        };

        let stability_multiplier = match self.metadata.stability {
            StabilityLevel::Stable => 1.0,
            StabilityLevel::Unstable => 0.8,
            StabilityLevel::Experimental => 0.6,
            StabilityLevel::Deprecated => 0.3,
            StabilityLevel::Internal => 0.2,
        };

        base_score * stability_multiplier * (1.0 + self.metadata.complexity / 100.0)
    }
}

/// Metadata for graph nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Documentation content
    pub documentation: Option<String>,
    /// Source file path
    pub source_file: Option<String>,
    /// Line number in source
    pub source_line: Option<u32>,
    /// Stability level
    pub stability: StabilityLevel,
    /// Complexity score (0-100)
    pub complexity: f64,
    /// Creation timestamp
    pub created_at: Option<DateTime<Utc>>,
    /// Last modification timestamp
    pub modified_at: Option<DateTime<Utc>>,
    /// Associated trait name (for implementations)
    pub trait_name: Option<String>,
    /// Generic parameters
    pub generic_parameters: Vec<String>,
    /// Where clauses
    pub where_clauses: Vec<String>,
    /// Deprecation information
    pub deprecation_note: Option<String>,
    /// Feature flags required
    pub feature_flags: Vec<String>,
    /// Module path
    pub module_path: Option<String>,
    /// Visibility (pub, pub(crate), etc.)
    pub visibility: Option<String>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            documentation: None,
            source_file: None,
            source_line: None,
            stability: StabilityLevel::Stable,
            complexity: 1.0,
            created_at: None,
            modified_at: None,
            trait_name: None,
            generic_parameters: Vec::new(),
            where_clauses: Vec::new(),
            deprecation_note: None,
            feature_flags: Vec::new(),
            module_path: None,
            visibility: None,
            attributes: HashMap::new(),
        }
    }
}

impl NodeMetadata {
    /// Check if this node is deprecated
    pub fn is_deprecated(&self) -> bool {
        self.stability == StabilityLevel::Deprecated || self.deprecation_note.is_some()
    }

    /// Check if this node is experimental
    pub fn is_experimental(&self) -> bool {
        self.stability == StabilityLevel::Experimental
    }

    /// Check if this node requires feature flags
    pub fn requires_features(&self) -> bool {
        !self.feature_flags.is_empty()
    }

    /// Get the age of this node in days
    pub fn age_in_days(&self) -> Option<i64> {
        self.created_at.map(|created| {
            let now = Utc::now();
            now.signed_duration_since(created).num_days()
        })
    }

    /// Get the time since last modification in days
    pub fn days_since_modification(&self) -> Option<i64> {
        self.modified_at.map(|modified| {
            let now = Utc::now();
            now.signed_duration_since(modified).num_days()
        })
    }
}

/// Edge in the trait graph with enhanced properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraitGraphEdge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Type of relationship
    pub edge_type: EdgeType,
    /// Weight/strength of the relationship
    pub weight: f64,
    /// Visual thickness for rendering
    pub thickness: Option<f64>,
    /// Visual color for rendering
    pub color: Option<String>,
    /// Display label for the edge
    pub label: Option<String>,
    /// Whether the edge is directed
    pub directed: bool,
    /// Additional metadata
    pub metadata: EdgeMetadata,
}

impl TraitGraphEdge {
    /// Create a new inheritance edge
    pub fn new_inheritance(from: String, to: String) -> Self {
        Self {
            from,
            to,
            edge_type: EdgeType::Inherits,
            weight: 1.0,
            thickness: Some(2.0),
            color: Some(EdgeType::Inherits.default_color().to_string()),
            label: Some(EdgeType::Inherits.display_name().to_string()),
            directed: true,
            metadata: EdgeMetadata::default(),
        }
    }

    /// Create a new implementation edge
    pub fn new_implementation(from: String, to: String) -> Self {
        Self {
            from,
            to,
            edge_type: EdgeType::Implements,
            weight: 0.8,
            thickness: Some(1.5),
            color: Some(EdgeType::Implements.default_color().to_string()),
            label: Some(EdgeType::Implements.display_name().to_string()),
            directed: true,
            metadata: EdgeMetadata::default(),
        }
    }

    /// Create a new usage edge
    pub fn new_usage(from: String, to: String) -> Self {
        Self {
            from,
            to,
            edge_type: EdgeType::Uses,
            weight: 0.5,
            thickness: Some(1.0),
            color: Some(EdgeType::Uses.default_color().to_string()),
            label: Some(EdgeType::Uses.display_name().to_string()),
            directed: false,
            metadata: EdgeMetadata::default(),
        }
    }

    /// Set the weight of the edge
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Set the visual thickness
    pub fn with_thickness(mut self, thickness: f64) -> Self {
        self.thickness = Some(thickness);
        self
    }

    /// Set the confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.metadata.confidence = confidence;
        self
    }

    /// Mark as conditional with conditions
    pub fn with_conditions(mut self, conditions: Vec<String>) -> Self {
        self.metadata.conditional = !conditions.is_empty();
        self.metadata.conditions = conditions;
        self
    }

    /// Get the reverse edge (swap from and to)
    pub fn reverse(&self) -> Self {
        Self {
            from: self.to.clone(),
            to: self.from.clone(),
            edge_type: self.edge_type,
            weight: self.weight,
            thickness: self.thickness,
            color: self.color.clone(),
            label: self.label.clone(),
            directed: self.directed,
            metadata: self.metadata.clone(),
        }
    }

    /// Check if this edge connects the given nodes (in either direction)
    pub fn connects(&self, node1: &str, node2: &str) -> bool {
        (self.from == node1 && self.to == node2) || (!self.directed && self.from == node2 && self.to == node1)
    }

    /// Get the other endpoint given one endpoint
    pub fn other_endpoint(&self, node: &str) -> Option<&str> {
        if self.from == node {
            Some(&self.to)
        } else if !self.directed && self.to == node {
            Some(&self.from)
        } else {
            None
        }
    }
}

/// Metadata for graph edges
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeMetadata {
    /// Confidence level of the relationship (0.0-1.0)
    pub confidence: f64,
    /// Source of the relationship information
    pub source: String,
    /// Line number where relationship is defined
    pub source_line: Option<u32>,
    /// Whether the relationship is conditional
    pub conditional: bool,
    /// Conditions under which the relationship exists
    pub conditions: Vec<String>,
    /// Feature flags affecting this relationship
    pub feature_flags: Vec<String>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

impl Default for EdgeMetadata {
    fn default() -> Self {
        Self {
            confidence: 1.0,
            source: "unknown".to_string(),
            source_line: None,
            conditional: false,
            conditions: Vec::new(),
            feature_flags: Vec::new(),
            attributes: HashMap::new(),
        }
    }
}

/// Metadata for the entire graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraitGraphMetadata {
    /// Title of the graph
    pub title: String,
    /// Description of what the graph represents
    pub description: Option<String>,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Version of the visualization system
    pub generator_version: String,
    /// Source crate or project name
    pub source_project: Option<String>,
    /// Git commit hash if available
    pub git_commit: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom metadata
    pub custom_metadata: HashMap<String, String>,
}

impl Default for TraitGraphMetadata {
    fn default() -> Self {
        Self {
            title: "Trait Relationship Graph".to_string(),
            description: None,
            generated_at: Utc::now(),
            generator_version: env!("CARGO_PKG_VERSION").to_string(),
            source_project: None,
            git_commit: None,
            tags: Vec::new(),
            custom_metadata: HashMap::new(),
        }
    }
}

/// Statistics about the graph structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Number of nodes by type
    pub nodes_by_type: HashMap<TraitNodeType, usize>,
    /// Number of edges by type
    pub edges_by_type: HashMap<EdgeType, usize>,
    /// Maximum depth of the graph
    pub max_depth: usize,
    /// Average degree (connections per node)
    pub average_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Average path length
    pub average_path_length: f64,
    /// Graph density
    pub density: f64,
    /// Number of connected components
    pub connected_components: usize,
    /// Diameter of the graph
    pub diameter: usize,
    /// Radius of the graph
    pub radius: usize,
}

impl Default for GraphStatistics {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            nodes_by_type: HashMap::new(),
            edges_by_type: HashMap::new(),
            max_depth: 0,
            average_degree: 0.0,
            clustering_coefficient: 0.0,
            average_path_length: 0.0,
            density: 0.0,
            connected_components: 0,
            diameter: 0,
            radius: 0,
        }
    }
}

impl GraphStatistics {
    /// Calculate basic statistics from nodes and edges
    pub fn from_graph(nodes: &[TraitGraphNode], edges: &[TraitGraphEdge]) -> Self {
        let mut stats = Self::default();

        stats.node_count = nodes.len();
        stats.edge_count = edges.len();

        // Count nodes by type
        for node in nodes {
            *stats.nodes_by_type.entry(node.node_type).or_insert(0) += 1;
        }

        // Count edges by type
        for edge in edges {
            *stats.edges_by_type.entry(edge.edge_type).or_insert(0) += 1;
        }

        // Calculate average degree
        if stats.node_count > 0 {
            stats.average_degree = (stats.edge_count * 2) as f64 / stats.node_count as f64;
        }

        // Calculate density
        if stats.node_count > 1 {
            let max_edges = stats.node_count * (stats.node_count - 1) / 2;
            stats.density = stats.edge_count as f64 / max_edges as f64;
        }

        stats
    }

    /// Check if the graph is sparse
    pub fn is_sparse(&self) -> bool {
        self.density < 0.1
    }

    /// Check if the graph is dense
    pub fn is_dense(&self) -> bool {
        self.density > 0.5
    }

    /// Get a summary description of the graph
    pub fn summary(&self) -> String {
        format!(
            "Graph with {} nodes, {} edges, density: {:.3}, avg degree: {:.2}",
            self.node_count, self.edge_count, self.density, self.average_degree
        )
    }
}

/// Results of graph analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphAnalysisResult {
    /// Centrality measures for each node
    pub centrality_measures: HashMap<String, CentralityMeasures>,
    /// Detected communities
    pub communities: Vec<Community>,
    /// Important paths in the graph
    pub critical_paths: Vec<GraphPath>,
    /// Graph quality metrics
    pub quality_metrics: GraphQualityMetrics,
    /// Analysis timestamp
    pub analyzed_at: DateTime<Utc>,
}

/// A community of related nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Community {
    /// Unique identifier for the community
    pub id: String,
    /// Nodes in this community
    pub nodes: Vec<String>,
    /// Modularity score of the community
    pub modularity: f64,
    /// Description of the community
    pub description: Option<String>,
}

impl Community {
    /// Create a new community
    pub fn new(id: String, nodes: Vec<String>) -> Self {
        Self {
            id,
            nodes,
            modularity: 0.0,
            description: None,
        }
    }

    /// Get the size of the community
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the community contains a specific node
    pub fn contains_node(&self, node_id: &str) -> bool {
        self.nodes.contains(&node_id.to_string())
    }

    /// Get the density of connections within the community
    pub fn internal_density(&self, edges: &[TraitGraphEdge]) -> f64 {
        let node_set: HashSet<_> = self.nodes.iter().collect();
        let internal_edges = edges.iter().filter(|edge| {
            node_set.contains(&edge.from) && node_set.contains(&edge.to)
        }).count();

        let max_edges = self.nodes.len() * (self.nodes.len() - 1) / 2;
        if max_edges > 0 {
            internal_edges as f64 / max_edges as f64
        } else {
            0.0
        }
    }
}

/// A path through the graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphPath {
    /// Nodes in the path
    pub nodes: Vec<String>,
    /// Total length of the path
    pub length: usize,
    /// Total weight of the path
    pub weight: f64,
}

impl GraphPath {
    /// Create a new path
    pub fn new(nodes: Vec<String>) -> Self {
        let length = nodes.len().saturating_sub(1);
        Self {
            nodes,
            length,
            weight: 0.0,
        }
    }

    /// Check if the path is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the start node
    pub fn start(&self) -> Option<&str> {
        self.nodes.first().map(|s| s.as_str())
    }

    /// Get the end node
    pub fn end(&self) -> Option<&str> {
        self.nodes.last().map(|s| s.as_str())
    }

    /// Check if the path contains a specific node
    pub fn contains_node(&self, node_id: &str) -> bool {
        self.nodes.contains(&node_id.to_string())
    }
}

/// Performance metrics for graph generation and analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Time taken for graph generation
    pub generation_time: Duration,
    /// Time taken for layout computation
    pub layout_time: Duration,
    /// Time taken for analysis
    pub analysis_time: Duration,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Number of layout iterations performed
    pub layout_iterations: u32,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Whether GPU acceleration was used
    pub gpu_accelerated: bool,
    /// Whether SIMD optimization was used
    pub simd_optimized: bool,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            generation_time: Duration::from_secs(0),
            layout_time: Duration::from_secs(0),
            analysis_time: Duration::from_secs(0),
            memory_usage: 0,
            layout_iterations: 0,
            cpu_utilization: 0.0,
            gpu_accelerated: false,
            simd_optimized: false,
        }
    }
}

impl PerformanceMetrics {
    /// Get the total processing time
    pub fn total_time(&self) -> Duration {
        self.generation_time + self.layout_time + self.analysis_time
    }

    /// Get a performance summary string
    pub fn summary(&self) -> String {
        format!(
            "Total: {:.2}s (Gen: {:.2}s, Layout: {:.2}s, Analysis: {:.2}s), Memory: {:.1}MB",
            self.total_time().as_secs_f64(),
            self.generation_time.as_secs_f64(),
            self.layout_time.as_secs_f64(),
            self.analysis_time.as_secs_f64(),
            self.memory_usage as f64 / 1024.0 / 1024.0
        )
    }

    /// Check if performance is acceptable
    pub fn is_acceptable(&self) -> bool {
        self.total_time() < Duration::from_secs(30) && self.memory_usage < 1024 * 1024 * 1024 // 1GB
    }
}

/// Centrality measures for graph analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CentralityMeasures {
    /// Degree centrality (0.0-1.0)
    pub degree: f64,
    /// Betweenness centrality (0.0-1.0)
    pub betweenness: f64,
    /// Closeness centrality (0.0-1.0)
    pub closeness: f64,
    /// Eigenvector centrality (0.0-1.0)
    pub eigenvector: f64,
    /// PageRank score (0.0-1.0)
    pub pagerank: f64,
}

impl Default for CentralityMeasures {
    fn default() -> Self {
        Self {
            degree: 0.0,
            betweenness: 0.0,
            closeness: 0.0,
            eigenvector: 0.0,
            pagerank: 0.0,
        }
    }
}

impl CentralityMeasures {
    /// Get the overall importance score (weighted average)
    pub fn importance_score(&self) -> f64 {
        (self.degree * 0.2 + self.betweenness * 0.3 + self.closeness * 0.2 + self.eigenvector * 0.2 + self.pagerank * 0.1)
    }

    /// Get the most important centrality measure
    pub fn dominant_measure(&self) -> &'static str {
        let measures = [
            ("degree", self.degree),
            ("betweenness", self.betweenness),
            ("closeness", self.closeness),
            ("eigenvector", self.eigenvector),
            ("pagerank", self.pagerank),
        ];

        measures.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| *name)
            .unwrap_or("degree")
    }
}

/// Quality metrics for graph visualization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphQualityMetrics {
    /// Visual clarity score (0.0-1.0)
    pub clarity: f64,
    /// Layout quality score (0.0-1.0)
    pub layout_quality: f64,
    /// Information density score (0.0-1.0)
    pub information_density: f64,
    /// Aesthetic appeal score (0.0-1.0)
    pub aesthetic_appeal: f64,
    /// Usability score (0.0-1.0)
    pub usability: f64,
}

impl Default for GraphQualityMetrics {
    fn default() -> Self {
        Self {
            clarity: 0.5,
            layout_quality: 0.5,
            information_density: 0.5,
            aesthetic_appeal: 0.5,
            usability: 0.5,
        }
    }
}

impl GraphQualityMetrics {
    /// Get the overall quality score
    pub fn overall_quality(&self) -> f64 {
        (self.clarity + self.layout_quality + self.information_density + self.aesthetic_appeal + self.usability) / 5.0
    }

    /// Check if the quality is acceptable
    pub fn is_acceptable(&self) -> bool {
        self.overall_quality() >= 0.6
    }
}

/// Main graph structure containing nodes, edges, and metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraitGraph {
    /// Nodes in the graph
    pub nodes: Vec<TraitGraphNode>,
    /// Edges in the graph
    pub edges: Vec<TraitGraphEdge>,
    /// Graph metadata
    pub metadata: TraitGraphMetadata,
    /// Graph statistics
    pub statistics: GraphStatistics,
    /// Performance metrics from generation
    pub performance: PerformanceMetrics,
}

impl TraitGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: TraitGraphMetadata::default(),
            statistics: GraphStatistics::default(),
            performance: PerformanceMetrics::default(),
        }
    }

    /// Create a graph with the given nodes and edges
    pub fn with_nodes_and_edges(nodes: Vec<TraitGraphNode>, edges: Vec<TraitGraphEdge>) -> Self {
        let statistics = GraphStatistics::from_graph(&nodes, &edges);
        Self {
            nodes,
            edges,
            metadata: TraitGraphMetadata::default(),
            statistics,
            performance: PerformanceMetrics::default(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: TraitGraphNode) {
        self.nodes.push(node);
        self.update_statistics();
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: TraitGraphEdge) {
        self.edges.push(edge);
        self.update_statistics();
    }

    /// Remove a node and all connected edges
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        let initial_len = self.nodes.len();
        self.nodes.retain(|n| n.id != node_id);

        if self.nodes.len() < initial_len {
            self.edges.retain(|e| e.from != node_id && e.to != node_id);
            self.update_statistics();
            true
        } else {
            false
        }
    }

    /// Find a node by ID
    pub fn find_node(&self, node_id: &str) -> Option<&TraitGraphNode> {
        self.nodes.iter().find(|n| n.id == node_id)
    }

    /// Find a node by ID (mutable)
    pub fn find_node_mut(&mut self, node_id: &str) -> Option<&mut TraitGraphNode> {
        self.nodes.iter_mut().find(|n| n.id == node_id)
    }

    /// Get all neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> Vec<&str> {
        let mut neighbors = Vec::new();
        for edge in &self.edges {
            if edge.from == node_id {
                neighbors.push(&edge.to);
            } else if !edge.directed && edge.to == node_id {
                neighbors.push(&edge.from);
            }
        }
        neighbors
    }

    /// Get all edges connected to a node
    pub fn get_node_edges(&self, node_id: &str) -> Vec<&TraitGraphEdge> {
        self.edges.iter().filter(|e| e.from == node_id || e.to == node_id).collect()
    }

    /// Check if two nodes are directly connected
    pub fn are_connected(&self, node1: &str, node2: &str) -> bool {
        self.edges.iter().any(|e| e.connects(node1, node2))
    }

    /// Get the degree of a node
    pub fn get_degree(&self, node_id: &str) -> usize {
        self.edges.iter().filter(|e| e.from == node_id || (!e.directed && e.to == node_id)).count()
    }

    /// Update graph statistics
    pub fn update_statistics(&mut self) {
        self.statistics = GraphStatistics::from_graph(&self.nodes, &self.edges);
    }

    /// Filter nodes by type
    pub fn nodes_by_type(&self, node_type: TraitNodeType) -> Vec<&TraitGraphNode> {
        self.nodes.iter().filter(|n| n.node_type == node_type).collect()
    }

    /// Filter edges by type
    pub fn edges_by_type(&self, edge_type: EdgeType) -> Vec<&TraitGraphEdge> {
        self.edges.iter().filter(|e| e.edge_type == edge_type).collect()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the total number of elements (nodes + edges)
    pub fn total_elements(&self) -> usize {
        self.nodes.len() + self.edges.len()
    }

    /// Validate the graph structure
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate node IDs
        let mut node_ids = HashSet::new();
        for node in &self.nodes {
            if !node_ids.insert(&node.id) {
                return Err(SklearsError::ValidationError(
                    format!("Duplicate node ID: {}", node.id)
                ));
            }
        }

        // Check that all edges reference existing nodes
        for edge in &self.edges {
            if !node_ids.contains(&edge.from) {
                return Err(SklearsError::ValidationError(
                    format!("Edge references non-existent source node: {}", edge.from)
                ));
            }
            if !node_ids.contains(&edge.to) {
                return Err(SklearsError::ValidationError(
                    format!("Edge references non-existent target node: {}", edge.to)
                ));
            }
        }

        Ok(())
    }
}

impl Default for TraitGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_graph_node_creation() {
        let node = TraitGraphNode::new_trait("test_trait".to_string(), "TestTrait".to_string());
        assert_eq!(node.id, "test_trait");
        assert_eq!(node.label, "TestTrait");
        assert_eq!(node.node_type, TraitNodeType::Trait);
        assert!(node.is_trait());
        assert!(!node.is_implementation());
    }

    #[test]
    fn test_trait_graph_edge_creation() {
        let edge = TraitGraphEdge::new_inheritance("trait1".to_string(), "trait2".to_string());
        assert_eq!(edge.from, "trait1");
        assert_eq!(edge.to, "trait2");
        assert_eq!(edge.edge_type, EdgeType::Inherits);
        assert!(edge.directed);
    }

    #[test]
    fn test_graph_creation_and_manipulation() {
        let mut graph = TraitGraph::new();
        assert!(graph.is_empty());

        let node = TraitGraphNode::new_trait("test".to_string(), "Test".to_string());
        graph.add_node(node);
        assert_eq!(graph.nodes.len(), 1);
        assert!(!graph.is_empty());

        let found = graph.find_node("test");
        assert!(found.is_some());
        assert_eq!(found.expect("expected valid value").label, "Test");
    }

    #[test]
    fn test_graph_statistics() {
        let nodes = vec![
            TraitGraphNode::new_trait("t1".to_string(), "Trait1".to_string()),
            TraitGraphNode::new_implementation("i1".to_string(), "Impl1".to_string(), "t1".to_string()),
        ];
        let edges = vec![
            TraitGraphEdge::new_implementation("t1".to_string(), "i1".to_string()),
        ];

        let stats = GraphStatistics::from_graph(&nodes, &edges);
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
        assert_eq!(stats.average_degree, 1.0);
    }

    #[test]
    fn test_community_operations() {
        let community = Community::new("c1".to_string(), vec!["n1".to_string(), "n2".to_string()]);
        assert_eq!(community.size(), 2);
        assert!(community.contains_node("n1"));
        assert!(!community.contains_node("n3"));
    }

    #[test]
    fn test_centrality_measures() {
        let measures = CentralityMeasures {
            degree: 0.8,
            betweenness: 0.6,
            closeness: 0.7,
            eigenvector: 0.5,
            pagerank: 0.9,
        };

        assert_eq!(measures.dominant_measure(), "pagerank");
        assert!(measures.importance_score() > 0.0);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            generation_time: Duration::from_secs(1),
            layout_time: Duration::from_secs(2),
            analysis_time: Duration::from_secs(1),
            memory_usage: 1024 * 1024, // 1MB
            ..Default::default()
        };

        assert_eq!(metrics.total_time(), Duration::from_secs(4));
        assert!(metrics.is_acceptable());
    }

    #[test]
    fn test_graph_validation() {
        let mut graph = TraitGraph::new();
        graph.add_node(TraitGraphNode::new_trait("t1".to_string(), "Trait1".to_string()));
        graph.add_edge(TraitGraphEdge::new_implementation("t1".to_string(), "t2".to_string()));

        // Should fail because t2 doesn't exist
        assert!(graph.validate().is_err());

        graph.add_node(TraitGraphNode::new_implementation("t2".to_string(), "Impl1".to_string(), "t1".to_string()));
        // Should pass now
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_node_metadata() {
        let metadata = NodeMetadata {
            stability: StabilityLevel::Deprecated,
            deprecation_note: Some("Use NewTrait instead".to_string()),
            ..Default::default()
        };

        assert!(metadata.is_deprecated());
        assert!(!metadata.is_experimental());
    }

    #[test]
    fn test_graph_navigation() {
        let mut graph = TraitGraph::new();
        graph.add_node(TraitGraphNode::new_trait("t1".to_string(), "Trait1".to_string()));
        graph.add_node(TraitGraphNode::new_trait("t2".to_string(), "Trait2".to_string()));
        graph.add_edge(TraitGraphEdge::new_inheritance("t1".to_string(), "t2".to_string()));

        let neighbors = graph.get_neighbors("t1");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], "t2");

        assert!(graph.are_connected("t1", "t2"));
        assert!(!graph.are_connected("t1", "nonexistent"));
    }
}