//! # Lineage Tracking Module
//!
//! Comprehensive lineage and provenance management system for tracking data flow,
//! transformations, and dependencies in machine learning workflows.
//!
//! ## Features
//!
//! - **Data Lineage**: Track complete data transformation chains
//! - **Provenance Management**: Record origin and history of data and models
//! - **Dependency Graphs**: Build and maintain complex dependency relationships
//! - **Audit Trails**: Comprehensive logging for regulatory compliance
//! - **Version Tracking**: Track versions and changes across time
//! - **Impact Analysis**: Analyze downstream impacts of changes
//! - **Performance Monitoring**: Comprehensive metrics for lineage operations
//!
//! ## Architecture
//!
//! ```text
//! LineageTracker
//! ├── LineageGraph (dependency graph management)
//! ├── ProvenanceRecorder (origin and history tracking)
//! ├── VersionManager (version control and tracking)
//! ├── AuditLogger (compliance and audit trails)
//! ├── ImpactAnalyzer (change impact analysis)
//! └── MetricsCollector (performance and health monitoring)
//! ```

use scirs2_core::error::{CoreError, Result};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::ndarray::{Array, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Lineage node representing a data or computation entity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LineageNode {
    /// Unique identifier for the node
    pub id: String,
    /// Node type classification
    pub node_type: NodeType,
    /// Display name for the node
    pub name: String,
    /// Description of the node
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
    /// Version identifier
    pub version: String,
    /// Tags for categorization
    pub tags: HashSet<String>,
    /// Custom attributes
    pub attributes: HashMap<String, serde_json::Value>,
    /// Status of the node
    pub status: NodeStatus,
}

/// Types of lineage nodes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// Raw data source
    DataSource,
    /// Transformed dataset
    Dataset,
    /// Machine learning model
    Model,
    /// Algorithm or transformation
    Algorithm,
    /// Execution or process
    Execution,
    /// Output or result
    Output,
    /// Custom node type
    Custom(String),
}

/// Status of lineage nodes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is active and current
    Active,
    /// Node is deprecated but still accessible
    Deprecated,
    /// Node has been archived
    Archived,
    /// Node has been deleted
    Deleted,
    /// Node is in error state
    Error(String),
}

/// Lineage edge representing a relationship between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEdge {
    /// Unique identifier for the edge
    pub id: String,
    /// Source node ID
    pub source_id: String,
    /// Target node ID
    pub target_id: String,
    /// Relationship type
    pub relationship: RelationshipType,
    /// Edge weight or strength
    pub weight: f64,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Custom attributes
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Types of relationships between nodes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Data dependency (A depends on B)
    DependsOn,
    /// Data transformation (A transforms to B)
    TransformsTo,
    /// Model training (A trains B)
    Trains,
    /// Model inference (A uses B for inference)
    InferenceUsing,
    /// Data derivation (A derives from B)
    DerivedFrom,
    /// Process execution (A executes B)
    Executes,
    /// Custom relationship type
    Custom(String),
}

/// Provenance record for tracking data origin and history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// Unique identifier
    pub id: String,
    /// Node this provenance refers to
    pub node_id: String,
    /// Origin information
    pub origin: OriginInfo,
    /// Transformation history
    pub transformations: Vec<TransformationStep>,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
    /// Validation status
    pub validation_status: ValidationStatus,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last verification timestamp
    pub last_verified: Option<SystemTime>,
}

/// Origin information for provenance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginInfo {
    /// Original source identifier
    pub source_id: String,
    /// Source type
    pub source_type: String,
    /// Location or path
    pub location: Option<String>,
    /// Creation method
    pub creation_method: String,
    /// Creator information
    pub creator: Option<String>,
    /// Original timestamp
    pub original_timestamp: SystemTime,
    /// Checksums or hashes
    pub checksums: HashMap<String, String>,
}

/// Transformation step in provenance history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationStep {
    /// Step identifier
    pub id: String,
    /// Transformation type
    pub transformation_type: String,
    /// Input references
    pub inputs: Vec<String>,
    /// Output references
    pub outputs: Vec<String>,
    /// Parameters used
    pub parameters: HashMap<String, serde_json::Value>,
    /// Transformation timestamp
    pub timestamp: SystemTime,
    /// Duration of transformation
    pub duration: Option<Duration>,
    /// Status of transformation
    pub status: TransformationStatus,
}

/// Status of transformation steps
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformationStatus {
    /// Transformation completed successfully
    Success,
    /// Transformation failed
    Failed(String),
    /// Transformation was skipped
    Skipped,
    /// Transformation is in progress
    InProgress,
}

/// Validation status for provenance records
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Fully validated and verified
    Validated,
    /// Partially validated with warnings
    PartiallyValidated(Vec<String>),
    /// Validation failed
    ValidationFailed(Vec<String>),
    /// Not yet validated
    NotValidated,
}

/// Lineage graph for managing dependencies and relationships
#[derive(Debug)]
pub struct LineageGraph {
    /// Nodes in the graph
    nodes: HashMap<String, LineageNode>,
    /// Edges in the graph
    edges: HashMap<String, LineageEdge>,
    /// Adjacency list for efficient traversal (outgoing edges)
    adjacency_out: HashMap<String, HashSet<String>>,
    /// Reverse adjacency list (incoming edges)
    adjacency_in: HashMap<String, HashSet<String>>,
    /// Node index by type
    type_index: HashMap<NodeType, HashSet<String>>,
    /// Node index by tags
    tag_index: HashMap<String, HashSet<String>>,
}

/// Impact analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    /// Source node that changed
    pub source_node_id: String,
    /// Nodes directly affected
    pub directly_affected: Vec<String>,
    /// Nodes indirectly affected
    pub indirectly_affected: Vec<String>,
    /// Impact severity score (0-100)
    pub severity_score: u8,
    /// Estimated time to propagate changes
    pub propagation_time: Option<Duration>,
    /// Recommended actions
    pub recommendations: Vec<String>,
    /// Analysis timestamp
    pub analyzed_at: SystemTime,
}

/// Lineage query for advanced graph traversal
#[derive(Debug, Clone)]
pub struct LineageQuery {
    /// Starting node ID
    pub start_node: String,
    /// Traversal direction
    pub direction: TraversalDirection,
    /// Maximum depth to traverse
    pub max_depth: Option<usize>,
    /// Filter by node types
    pub node_types: Option<HashSet<NodeType>>,
    /// Filter by relationship types
    pub relationship_types: Option<HashSet<RelationshipType>>,
    /// Filter by node status
    pub node_status: Option<HashSet<NodeStatus>>,
    /// Include inactive nodes
    pub include_inactive: bool,
}

/// Direction for graph traversal
#[derive(Debug, Clone, PartialEq)]
pub enum TraversalDirection {
    /// Follow outgoing edges (forward)
    Forward,
    /// Follow incoming edges (backward)
    Backward,
    /// Follow both directions
    Bidirectional,
}

/// Lineage tracker configuration
#[derive(Debug, Clone)]
pub struct LineageConfig {
    /// Maximum nodes before cleanup
    pub max_nodes: usize,
    /// Maximum edges before cleanup
    pub max_edges: usize,
    /// Enable automatic validation
    pub auto_validation: bool,
    /// Validation interval
    pub validation_interval: Option<Duration>,
    /// Enable impact analysis
    pub enable_impact_analysis: bool,
    /// Retention period for deleted nodes
    pub retention_period: Duration,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
}

impl Default for LineageConfig {
    fn default() -> Self {
        Self {
            max_nodes: 50_000,
            max_edges: 200_000,
            auto_validation: true,
            validation_interval: Some(Duration::from_secs(3600)), // 1 hour
            enable_impact_analysis: true,
            retention_period: Duration::from_secs(86400 * 30), // 30 days
            enable_monitoring: true,
        }
    }
}

/// Main lineage tracker
#[derive(Debug)]
pub struct LineageTracker {
    /// Tracker configuration
    config: LineageConfig,
    /// Lineage graph
    graph: LineageGraph,
    /// Provenance records
    provenance_records: HashMap<String, ProvenanceRecord>,
    /// Audit trail
    audit_trail: Vec<AuditEntry>,
    /// Performance metrics
    metrics: Arc<MetricRegistry>,
    /// Operation timers
    add_node_timer: Timer,
    add_edge_timer: Timer,
    query_timer: Timer,
    validation_timer: Timer,
    /// Operation counters
    nodes_added: Counter,
    edges_added: Counter,
    queries_executed: Counter,
    validations_performed: Counter,
}

/// Audit entry for compliance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry ID
    pub id: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Operation type
    pub operation: String,
    /// User or system that performed the operation
    pub actor: String,
    /// Affected resources
    pub resources: Vec<String>,
    /// Operation details
    pub details: HashMap<String, serde_json::Value>,
    /// Operation result
    pub result: AuditResult,
}

/// Result of audited operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditResult {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failed(String),
    /// Operation was partial
    Partial(String),
}

impl LineageGraph {
    /// Create a new empty lineage graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
            type_index: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: LineageNode) -> Result<()> {
        let node_id = node.id.clone();

        // Update type index
        self.type_index
            .entry(node.node_type.clone())
            .or_insert_with(HashSet::new)
            .insert(node_id.clone());

        // Update tag index
        for tag in &node.tags {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(HashSet::new)
                .insert(node_id.clone());
        }

        // Initialize adjacency lists
        self.adjacency_out.entry(node_id.clone()).or_insert_with(HashSet::new);
        self.adjacency_in.entry(node_id.clone()).or_insert_with(HashSet::new);

        // Add node
        self.nodes.insert(node_id, node);

        Ok(())
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: LineageEdge) -> Result<()> {
        let edge_id = edge.id.clone();
        let source_id = edge.source_id.clone();
        let target_id = edge.target_id.clone();

        // Verify nodes exist
        if !self.nodes.contains_key(&source_id) {
            return Err(CoreError::ValidationError(
                format!("Source node {} not found", source_id)
            ));
        }

        if !self.nodes.contains_key(&target_id) {
            return Err(CoreError::ValidationError(
                format!("Target node {} not found", target_id)
            ));
        }

        // Update adjacency lists
        self.adjacency_out
            .entry(source_id.clone())
            .or_insert_with(HashSet::new)
            .insert(target_id.clone());

        self.adjacency_in
            .entry(target_id.clone())
            .or_insert_with(HashSet::new)
            .insert(source_id.clone());

        // Add edge
        self.edges.insert(edge_id, edge);

        Ok(())
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: &str) -> Option<&LineageNode> {
        self.nodes.get(node_id)
    }

    /// Get edge by ID
    pub fn get_edge(&self, edge_id: &str) -> Option<&LineageEdge> {
        self.edges.get(edge_id)
    }

    /// Get all nodes of a specific type
    pub fn get_nodes_by_type(&self, node_type: &NodeType) -> Vec<&LineageNode> {
        self.type_index
            .get(node_type)
            .map(|node_ids| {
                node_ids.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get nodes by tags
    pub fn get_nodes_by_tag(&self, tag: &str) -> Vec<&LineageNode> {
        self.tag_index
            .get(tag)
            .map(|node_ids| {
                node_ids.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get outgoing neighbors of a node
    pub fn get_outgoing_neighbors(&self, node_id: &str) -> Vec<&LineageNode> {
        self.adjacency_out
            .get(node_id)
            .map(|neighbors| {
                neighbors.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get incoming neighbors of a node
    pub fn get_incoming_neighbors(&self, node_id: &str) -> Vec<&LineageNode> {
        self.adjacency_in
            .get(node_id)
            .map(|neighbors| {
                neighbors.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Perform breadth-first traversal
    pub fn bfs_traversal(&self, query: &LineageQuery) -> Vec<LineageNode> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        queue.push_back((query.start_node.clone(), 0));
        visited.insert(query.start_node.clone());

        while let Some((current_id, depth)) = queue.pop_front() {
            // Check depth limit
            if let Some(max_depth) = query.max_depth {
                if depth >= max_depth {
                    continue;
                }
            }

            if let Some(node) = self.nodes.get(&current_id) {
                // Apply filters
                if self.node_matches_query_filters(node, query) {
                    result.push(node.clone());
                }

                // Get neighbors based on direction
                let neighbors = match query.direction {
                    TraversalDirection::Forward => {
                        self.adjacency_out.get(&current_id).cloned().unwrap_or_default()
                    }
                    TraversalDirection::Backward => {
                        self.adjacency_in.get(&current_id).cloned().unwrap_or_default()
                    }
                    TraversalDirection::Bidirectional => {
                        let mut all_neighbors = self.adjacency_out.get(&current_id).cloned().unwrap_or_default();
                        all_neighbors.extend(self.adjacency_in.get(&current_id).cloned().unwrap_or_default());
                        all_neighbors
                    }
                };

                // Add unvisited neighbors to queue
                for neighbor_id in neighbors {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id.clone());
                        queue.push_back((neighbor_id, depth + 1));
                    }
                }
            }
        }

        result
    }

    /// Check if node matches query filters
    fn node_matches_query_filters(&self, node: &LineageNode, query: &LineageQuery) -> bool {
        // Filter by node types
        if let Some(ref types) = query.node_types {
            if !types.contains(&node.node_type) {
                return false;
            }
        }

        // Filter by node status
        if let Some(ref statuses) = query.node_status {
            if !statuses.contains(&node.status) {
                return false;
            }
        }

        // Include inactive nodes check
        if !query.include_inactive {
            match node.status {
                NodeStatus::Active => {}
                _ => return false,
            }
        }

        true
    }

    /// Find cycles in the graph
    pub fn detect_cycles(&self) -> Vec<Vec<String>> {
        let mut visited = HashMap::new();
        let mut rec_stack = HashSet::new();
        let mut cycles = Vec::new();

        for node_id in self.nodes.keys() {
            if !visited.contains_key(node_id) {
                self.dfs_cycle_detection(
                    node_id,
                    &mut visited,
                    &mut rec_stack,
                    &mut cycles,
                    &mut vec![node_id.clone()]
                );
            }
        }

        cycles
    }

    fn dfs_cycle_detection(
        &self,
        node_id: &str,
        visited: &mut HashMap<String, bool>,
        rec_stack: &mut HashSet<String>,
        cycles: &mut Vec<Vec<String>>,
        path: &mut Vec<String>,
    ) {
        visited.insert(node_id.to_string(), true);
        rec_stack.insert(node_id.to_string());

        if let Some(neighbors) = self.adjacency_out.get(node_id) {
            for neighbor in neighbors {
                if !visited.get(neighbor).unwrap_or(&false) {
                    path.push(neighbor.clone());
                    self.dfs_cycle_detection(neighbor, visited, rec_stack, cycles, path);
                    path.pop();
                } else if rec_stack.contains(neighbor) {
                    // Found cycle
                    let cycle_start = path.iter().position(|x| x == neighbor).unwrap_or(0);
                    cycles.push(path[cycle_start..].to_vec());
                }
            }
        }

        rec_stack.remove(node_id);
    }

    /// Get graph statistics
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        stats.insert("total_nodes".to_string(), json!(self.nodes.len()));
        stats.insert("total_edges".to_string(), json!(self.edges.len()));

        // Node type distribution
        let mut type_counts = HashMap::new();
        for node_type in self.type_index.keys() {
            let count = self.type_index.get(node_type).map(|s| s.len()).unwrap_or(0);
            type_counts.insert(format!("{:?}", node_type), count);
        }
        stats.insert("node_type_distribution".to_string(), json!(type_counts));

        // Graph connectivity
        let connected_components = self.count_connected_components();
        stats.insert("connected_components".to_string(), json!(connected_components));

        stats
    }

    fn count_connected_components(&self) -> usize {
        let mut visited = HashSet::new();
        let mut components = 0;

        for node_id in self.nodes.keys() {
            if !visited.contains(node_id) {
                self.dfs_mark_component(node_id, &mut visited);
                components += 1;
            }
        }

        components
    }

    fn dfs_mark_component(&self, node_id: &str, visited: &mut HashSet<String>) {
        visited.insert(node_id.to_string());

        // Visit outgoing neighbors
        if let Some(neighbors) = self.adjacency_out.get(node_id) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    self.dfs_mark_component(neighbor, visited);
                }
            }
        }

        // Visit incoming neighbors
        if let Some(neighbors) = self.adjacency_in.get(node_id) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    self.dfs_mark_component(neighbor, visited);
                }
            }
        }
    }
}

impl LineageTracker {
    /// Create a new lineage tracker
    pub fn new() -> Self {
        Self::with_config(LineageConfig::default())
    }

    /// Create lineage tracker with configuration
    pub fn with_config(config: LineageConfig) -> Self {
        let metrics = Arc::new(MetricRegistry::new());

        Self {
            config,
            graph: LineageGraph::new(),
            provenance_records: HashMap::new(),
            audit_trail: Vec::new(),
            metrics: metrics.clone(),
            add_node_timer: metrics.timer("lineage.add_node_duration"),
            add_edge_timer: metrics.timer("lineage.add_edge_duration"),
            query_timer: metrics.timer("lineage.query_duration"),
            validation_timer: metrics.timer("lineage.validation_duration"),
            nodes_added: metrics.counter("lineage.nodes_added"),
            edges_added: metrics.counter("lineage.edges_added"),
            queries_executed: metrics.counter("lineage.queries_executed"),
            validations_performed: metrics.counter("lineage.validations_performed"),
        }
    }

    /// Add a node to the lineage graph
    pub fn add_node(
        &mut self,
        node_type: NodeType,
        name: String,
        description: Option<String>,
        tags: HashSet<String>,
        attributes: HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        let _timer = self.add_node_timer.start_timer();

        let node_id = Uuid::new_v4().to_string();
        let now = SystemTime::now();

        let node = LineageNode {
            id: node_id.clone(),
            node_type: node_type.clone(),
            name: name.clone(),
            description,
            created_at: now,
            updated_at: now,
            version: "1.0".to_string(),
            tags: tags.clone(),
            attributes: attributes.clone(),
            status: NodeStatus::Active,
        };

        self.graph.add_node(node)?;

        // Create audit entry
        self.add_audit_entry(
            "add_node".to_string(),
            "system".to_string(),
            vec![node_id.clone()],
            json!({
                "node_type": node_type,
                "name": name,
                "tags": tags.into_iter().collect::<Vec<_>>()
            }),
            AuditResult::Success,
        );

        self.nodes_added.inc();
        Ok(node_id)
    }

    /// Add an edge to the lineage graph
    pub fn add_edge(
        &mut self,
        source_id: String,
        target_id: String,
        relationship: RelationshipType,
        weight: f64,
        attributes: HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        let _timer = self.add_edge_timer.start_timer();

        let edge_id = Uuid::new_v4().to_string();
        let now = SystemTime::now();

        let edge = LineageEdge {
            id: edge_id.clone(),
            source_id: source_id.clone(),
            target_id: target_id.clone(),
            relationship: relationship.clone(),
            weight,
            created_at: now,
            attributes: attributes.clone(),
        };

        self.graph.add_edge(edge)?;

        // Create audit entry
        self.add_audit_entry(
            "add_edge".to_string(),
            "system".to_string(),
            vec![edge_id.clone(), source_id.clone(), target_id.clone()],
            json!({
                "relationship": relationship,
                "weight": weight
            }),
            AuditResult::Success,
        );

        self.edges_added.inc();
        Ok(edge_id)
    }

    /// Record provenance information
    pub fn record_provenance(
        &mut self,
        node_id: String,
        origin: OriginInfo,
        transformations: Vec<TransformationStep>,
        quality_metrics: HashMap<String, f64>,
    ) -> Result<String> {
        let provenance_id = Uuid::new_v4().to_string();
        let now = SystemTime::now();

        let provenance = ProvenanceRecord {
            id: provenance_id.clone(),
            node_id: node_id.clone(),
            origin,
            transformations,
            quality_metrics,
            validation_status: ValidationStatus::NotValidated,
            created_at: now,
            last_verified: None,
        };

        self.provenance_records.insert(provenance_id.clone(), provenance);

        // Create audit entry
        self.add_audit_entry(
            "record_provenance".to_string(),
            "system".to_string(),
            vec![node_id, provenance_id.clone()],
            json!({}),
            AuditResult::Success,
        );

        Ok(provenance_id)
    }

    /// Query lineage graph
    pub fn query_lineage(&mut self, query: LineageQuery) -> Result<Vec<LineageNode>> {
        let _timer = self.query_timer.start_timer();

        let results = self.graph.bfs_traversal(&query);

        self.queries_executed.inc();
        Ok(results)
    }

    /// Perform impact analysis for a node change
    pub fn analyze_impact(&self, node_id: &str) -> Result<ImpactAnalysis> {
        let mut directly_affected = Vec::new();
        let mut indirectly_affected = Vec::new();

        // Find directly affected nodes (immediate neighbors)
        let direct_neighbors = self.graph.get_outgoing_neighbors(node_id);
        for neighbor in direct_neighbors {
            directly_affected.push(neighbor.id.clone());
        }

        // Find indirectly affected nodes (2+ hops away)
        let mut visited = HashSet::new();
        visited.insert(node_id.to_string());

        for direct_id in &directly_affected {
            visited.insert(direct_id.clone());
            self.collect_indirect_impacts(direct_id, &mut indirectly_affected, &mut visited, 1, 3);
        }

        // Calculate severity score
        let total_affected = directly_affected.len() + indirectly_affected.len();
        let severity_score = std::cmp::min(100, (total_affected * 10) as u8);

        Ok(ImpactAnalysis {
            source_node_id: node_id.to_string(),
            directly_affected,
            indirectly_affected,
            severity_score,
            propagation_time: Some(Duration::from_secs(total_affected as u64 * 60)),
            recommendations: self.generate_impact_recommendations(severity_score, total_affected),
            analyzed_at: SystemTime::now(),
        })
    }

    /// Get provenance for a node
    pub fn get_provenance(&self, node_id: &str) -> Option<&ProvenanceRecord> {
        self.provenance_records.values()
            .find(|p| p.node_id == node_id)
    }

    /// Get lineage statistics
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = self.graph.get_statistics();

        stats.insert("total_provenance_records".to_string(),
                    json!(self.provenance_records.len()));
        stats.insert("total_audit_entries".to_string(),
                    json!(self.audit_trail.len()));

        // Performance metrics
        stats.insert("nodes_added_total".to_string(),
                    json!(self.nodes_added.get()));
        stats.insert("edges_added_total".to_string(),
                    json!(self.edges_added.get()));
        stats.insert("queries_executed_total".to_string(),
                    json!(self.queries_executed.get()));

        stats
    }

    /// Export lineage data
    pub fn export_lineage(&self) -> Result<String> {
        let export_data = serde_json::json!({
            "version": "1.0",
            "timestamp": SystemTime::now(),
            "nodes": self.graph.nodes.values().collect::<Vec<_>>(),
            "edges": self.graph.edges.values().collect::<Vec<_>>(),
            "provenance_records": self.provenance_records.values().collect::<Vec<_>>(),
            "audit_trail": self.audit_trail,
            "statistics": self.get_statistics()
        });

        serde_json::to_string_pretty(&export_data)
            .map_err(|e| CoreError::SerializationError(format!("Export failed: {}", e)))
    }

    // Private helper methods

    fn collect_indirect_impacts(
        &self,
        node_id: &str,
        indirect_affects: &mut Vec<String>,
        visited: &mut HashSet<String>,
        current_depth: usize,
        max_depth: usize,
    ) {
        if current_depth >= max_depth {
            return;
        }

        let neighbors = self.graph.get_outgoing_neighbors(node_id);
        for neighbor in neighbors {
            if !visited.contains(&neighbor.id) {
                visited.insert(neighbor.id.clone());
                indirect_affects.push(neighbor.id.clone());

                self.collect_indirect_impacts(
                    &neighbor.id,
                    indirect_affects,
                    visited,
                    current_depth + 1,
                    max_depth,
                );
            }
        }
    }

    fn generate_impact_recommendations(&self, severity_score: u8, total_affected: usize) -> Vec<String> {
        let mut recommendations = Vec::new();

        if severity_score > 80 {
            recommendations.push("HIGH IMPACT: Schedule maintenance window for updates".to_string());
            recommendations.push("Notify all stakeholders before making changes".to_string());
        } else if severity_score > 50 {
            recommendations.push("MEDIUM IMPACT: Test changes in staging environment".to_string());
            recommendations.push("Consider gradual rollout".to_string());
        } else {
            recommendations.push("LOW IMPACT: Standard change process applicable".to_string());
        }

        if total_affected > 100 {
            recommendations.push("Consider batch processing for efficiency".to_string());
        }

        recommendations
    }

    fn add_audit_entry(
        &mut self,
        operation: String,
        actor: String,
        resources: Vec<String>,
        details: serde_json::Value,
        result: AuditResult,
    ) {
        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            operation,
            actor,
            resources,
            details: details.as_object().unwrap_or(&serde_json::Map::new()).clone(),
            result,
        };

        self.audit_trail.push(entry);

        // Keep audit trail size manageable
        if self.audit_trail.len() > 10_000 {
            self.audit_trail.drain(0..1000); // Remove oldest 1000 entries
        }
    }
}

impl Default for LineageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_lineage_basic_operations() {
        let mut tracker = LineageTracker::new();

        // Add nodes
        let source_id = tracker.add_node(
            NodeType::DataSource,
            "Test Data".to_string(),
            Some("Test data source".to_string()),
            ["source", "test"].iter().map(|s| s.to_string()).collect(),
            HashMap::new(),
        ).unwrap();

        let model_id = tracker.add_node(
            NodeType::Model,
            "Test Model".to_string(),
            Some("Test ML model".to_string()),
            ["model", "test"].iter().map(|s| s.to_string()).collect(),
            HashMap::new(),
        ).unwrap();

        // Add edge
        let edge_id = tracker.add_edge(
            source_id.clone(),
            model_id.clone(),
            RelationshipType::Trains,
            1.0,
            HashMap::new(),
        ).unwrap();

        // Verify nodes exist
        assert!(tracker.graph.get_node(&source_id).is_some());
        assert!(tracker.graph.get_node(&model_id).is_some());
        assert!(tracker.graph.get_edge(&edge_id).is_some());

        // Test query
        let query = LineageQuery {
            start_node: source_id.clone(),
            direction: TraversalDirection::Forward,
            max_depth: Some(2),
            node_types: None,
            relationship_types: None,
            node_status: None,
            include_inactive: false,
        };

        let results = tracker.query_lineage(query).unwrap();
        assert_eq!(results.len(), 2); // Both nodes should be included
    }

    #[test]
    fn test_provenance_tracking() {
        let mut tracker = LineageTracker::new();

        let node_id = tracker.add_node(
            NodeType::Dataset,
            "Processed Data".to_string(),
            None,
            HashSet::new(),
            HashMap::new(),
        ).unwrap();

        let origin = OriginInfo {
            source_id: "original_data".to_string(),
            source_type: "CSV".to_string(),
            location: Some("/data/raw.csv".to_string()),
            creation_method: "file_upload".to_string(),
            creator: Some("data_scientist".to_string()),
            original_timestamp: SystemTime::now(),
            checksums: [("md5".to_string(), "abc123".to_string())].iter().cloned().collect(),
        };

        let transformations = vec![
            TransformationStep {
                id: Uuid::new_v4().to_string(),
                transformation_type: "normalization".to_string(),
                inputs: vec!["original_data".to_string()],
                outputs: vec![node_id.clone()],
                parameters: [("method".to_string(), json!("z-score"))].iter().cloned().collect(),
                timestamp: SystemTime::now(),
                duration: Some(Duration::from_secs(30)),
                status: TransformationStatus::Success,
            }
        ];

        let quality_metrics = [("completeness".to_string(), 0.95)].iter().cloned().collect();

        let provenance_id = tracker.record_provenance(
            node_id.clone(),
            origin,
            transformations,
            quality_metrics,
        ).unwrap();

        // Verify provenance was recorded
        let provenance = tracker.get_provenance(&node_id).unwrap();
        assert_eq!(provenance.node_id, node_id);
        assert_eq!(provenance.transformations.len(), 1);
        assert_eq!(provenance.quality_metrics["completeness"], 0.95);
    }

    #[test]
    fn test_impact_analysis() {
        let mut tracker = LineageTracker::new();

        // Create a chain of dependencies
        let node1 = tracker.add_node(
            NodeType::DataSource,
            "Source".to_string(),
            None,
            HashSet::new(),
            HashMap::new(),
        ).unwrap();

        let node2 = tracker.add_node(
            NodeType::Dataset,
            "Dataset".to_string(),
            None,
            HashSet::new(),
            HashMap::new(),
        ).unwrap();

        let node3 = tracker.add_node(
            NodeType::Model,
            "Model".to_string(),
            None,
            HashSet::new(),
            HashMap::new(),
        ).unwrap();

        // Add edges
        tracker.add_edge(
            node1.clone(),
            node2.clone(),
            RelationshipType::TransformsTo,
            1.0,
            HashMap::new(),
        ).unwrap();

        tracker.add_edge(
            node2.clone(),
            node3.clone(),
            RelationshipType::Trains,
            1.0,
            HashMap::new(),
        ).unwrap();

        // Analyze impact of changing node1
        let impact = tracker.analyze_impact(&node1).unwrap();

        assert_eq!(impact.source_node_id, node1);
        assert_eq!(impact.directly_affected.len(), 1);
        assert!(impact.directly_affected.contains(&node2));
        assert!(!impact.recommendations.is_empty());
    }

    #[test]
    fn test_cycle_detection() {
        let mut tracker = LineageTracker::new();

        // Create nodes
        let node1 = tracker.add_node(
            NodeType::Dataset,
            "Node1".to_string(),
            None,
            HashSet::new(),
            HashMap::new(),
        ).unwrap();

        let node2 = tracker.add_node(
            NodeType::Dataset,
            "Node2".to_string(),
            None,
            HashSet::new(),
            HashMap::new(),
        ).unwrap();

        let node3 = tracker.add_node(
            NodeType::Dataset,
            "Node3".to_string(),
            None,
            HashSet::new(),
            HashMap::new(),
        ).unwrap();

        // Create a cycle: node1 -> node2 -> node3 -> node1
        tracker.add_edge(
            node1.clone(),
            node2.clone(),
            RelationshipType::TransformsTo,
            1.0,
            HashMap::new(),
        ).unwrap();

        tracker.add_edge(
            node2.clone(),
            node3.clone(),
            RelationshipType::TransformsTo,
            1.0,
            HashMap::new(),
        ).unwrap();

        tracker.add_edge(
            node3.clone(),
            node1.clone(),
            RelationshipType::TransformsTo,
            1.0,
            HashMap::new(),
        ).unwrap();

        // Detect cycles
        let cycles = tracker.graph.detect_cycles();
        assert!(!cycles.is_empty());
    }
}