//! Audit trails and explanation lineage tracking
//!
//! This module provides comprehensive audit logging and lineage tracking for explanation systems,
//! enabling full traceability of explanation generation, modification, and access patterns.

use crate::{SklResult, SklearsError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Types of audit events that can be logged
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditEventType {
    // User authentication events
    /// UserLogin

    /// UserLogin
    UserLogin,
    /// UserLogout

    /// UserLogout
    UserLogout,
    /// UserFailedLogin

    /// UserFailedLogin
    UserFailedLogin,

    // Explanation operations
    /// ExplanationGenerated

    /// ExplanationGenerated
    ExplanationGenerated,
    /// ExplanationViewed

    /// ExplanationViewed
    ExplanationViewed,
    /// ExplanationModified

    /// ExplanationModified
    ExplanationModified,
    /// ExplanationDeleted

    /// ExplanationDeleted
    ExplanationDeleted,
    /// ExplanationExported
    ExplanationExported,

    // Model operations
    /// ModelRegistered
    ModelRegistered,
    /// ModelViewed
    ModelViewed,
    /// ModelModified
    ModelModified,
    /// ModelDeleted
    ModelDeleted,

    // Data operations
    /// DataUploaded
    DataUploaded,
    /// DataViewed
    DataViewed,
    /// DataModified
    DataModified,
    /// DataDeleted
    DataDeleted,

    // Administrative operations
    /// UserCreated
    UserCreated,
    /// UserModified
    UserModified,
    /// UserDeleted
    UserDeleted,
    /// RoleCreated
    RoleCreated,
    /// RoleModified
    RoleModified,
    /// RoleDeleted
    RoleDeleted,
    /// PermissionGranted
    PermissionGranted,
    /// PermissionRevoked
    PermissionRevoked,

    // System events
    /// SystemStartup
    SystemStartup,
    /// SystemShutdown
    SystemShutdown,
    /// ConfigurationChanged
    ConfigurationChanged,
    /// SecurityPolicyChanged
    SecurityPolicyChanged,

    // Error events
    /// AccessDenied
    AccessDenied,
    /// InvalidRequest
    InvalidRequest,
    /// SystemError
    SystemError,
}

impl AuditEventType {
    /// Get the severity level of the event type
    pub fn severity(&self) -> AuditSeverity {
        match self {
            AuditEventType::UserFailedLogin
            | AuditEventType::AccessDenied
            | AuditEventType::InvalidRequest => AuditSeverity::Warning,

            AuditEventType::SystemError
            | AuditEventType::UserDeleted
            | AuditEventType::ModelDeleted
            | AuditEventType::DataDeleted => AuditSeverity::Error,

            AuditEventType::SystemStartup
            | AuditEventType::SystemShutdown
            | AuditEventType::SecurityPolicyChanged => AuditSeverity::Critical,

            _ => AuditSeverity::Info,
        }
    }

    /// Check if this event type should be retained for compliance
    pub fn requires_long_term_retention(&self) -> bool {
        matches!(
            self,
            AuditEventType::ExplanationGenerated
                | AuditEventType::ExplanationModified
                | AuditEventType::ExplanationDeleted
                | AuditEventType::ModelRegistered
                | AuditEventType::ModelModified
                | AuditEventType::ModelDeleted
                | AuditEventType::DataUploaded
                | AuditEventType::DataModified
                | AuditEventType::DataDeleted
                | AuditEventType::UserCreated
                | AuditEventType::UserModified
                | AuditEventType::UserDeleted
                | AuditEventType::RoleCreated
                | AuditEventType::RoleModified
                | AuditEventType::RoleDeleted
                | AuditEventType::SecurityPolicyChanged
        )
    }
}

/// Severity levels for audit events
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditSeverity {
    /// Info

    /// Info
    Info,
    /// Warning

    /// Warning
    Warning,
    /// Error

    /// Error
    Error,
    /// Critical

    /// Critical
    Critical,
}

/// Types of operations that can be tracked for lineage
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    /// DataIngestion

    /// DataIngestion
    DataIngestion,
    /// DataTransformation

    /// DataTransformation
    DataTransformation,
    /// ModelTraining

    /// ModelTraining
    ModelTraining,
    /// ModelInference

    /// ModelInference
    ModelInference,
    /// ExplanationGeneration

    /// ExplanationGeneration
    ExplanationGeneration,
    /// ExplanationTransformation

    /// ExplanationTransformation
    ExplanationTransformation,
    /// ExplanationAggregation

    /// ExplanationAggregation
    ExplanationAggregation,
}

/// Individual audit event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID
    pub id: String,
    /// Type of event
    pub event_type: AuditEventType,
    /// Event severity
    pub severity: AuditSeverity,
    /// Timestamp when the event occurred
    pub timestamp: DateTime<Utc>,
    /// User who triggered the event
    pub user_id: Option<String>,
    /// Session ID if applicable
    pub session_id: Option<String>,
    /// Resource affected by the event
    pub resource_id: Option<String>,
    /// Resource type (model, explanation, user, etc.)
    pub resource_type: Option<String>,
    /// Operation performed
    pub operation: String,
    /// Event description
    pub description: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// IP address of the client
    pub client_ip: Option<String>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Whether the operation was successful
    pub success: bool,
    /// Error message if operation failed
    pub error_message: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event
    pub fn new(event_type: AuditEventType, operation: String, description: String) -> Self {
        let severity = event_type.severity();

        Self {
            id: Uuid::new_v4().to_string(),
            event_type,
            severity,
            timestamp: Utc::now(),
            user_id: None,
            session_id: None,
            resource_id: None,
            resource_type: None,
            operation,
            description,
            metadata: HashMap::new(),
            client_ip: None,
            user_agent: None,
            success: true,
            error_message: None,
        }
    }

    /// Set user information
    pub fn with_user(mut self, user_id: String, session_id: Option<String>) -> Self {
        self.user_id = Some(user_id);
        self.session_id = session_id;
        self
    }

    /// Set resource information
    pub fn with_resource(mut self, resource_id: String, resource_type: String) -> Self {
        self.resource_id = Some(resource_id);
        self.resource_type = Some(resource_type);
        self
    }

    /// Set client information
    pub fn with_client_info(
        mut self,
        client_ip: Option<String>,
        user_agent: Option<String>,
    ) -> Self {
        self.client_ip = client_ip;
        self.user_agent = user_agent;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Mark as failed operation
    pub fn with_error(mut self, error_message: String) -> Self {
        self.success = false;
        self.error_message = Some(error_message);
        self
    }
}

/// Audit record containing multiple related events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    /// Record ID
    pub id: String,
    /// Transaction or operation ID that groups related events
    pub transaction_id: String,
    /// All events in this record
    pub events: Vec<AuditEvent>,
    /// Start time of the transaction
    pub start_time: DateTime<Utc>,
    /// End time of the transaction
    pub end_time: Option<DateTime<Utc>>,
    /// Overall success status
    pub success: bool,
}

impl AuditRecord {
    /// Create a new audit record
    pub fn new(transaction_id: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            transaction_id,
            events: Vec::new(),
            start_time: Utc::now(),
            end_time: None,
            success: true,
        }
    }

    /// Add an event to the record
    pub fn add_event(&mut self, event: AuditEvent) {
        if !event.success {
            self.success = false;
        }
        self.events.push(event);
    }

    /// Mark the record as completed
    pub fn complete(&mut self) {
        self.end_time = Some(Utc::now());
    }

    /// Get the duration of the transaction
    pub fn duration(&self) -> Option<chrono::Duration> {
        self.end_time.map(|end| end - self.start_time)
    }
}

/// Full audit trail containing all audit records
#[derive(Debug)]
pub struct AuditTrail {
    /// All audit records
    records: VecDeque<AuditRecord>,
    /// Maximum number of records to keep in memory
    max_records: usize,
    /// Total number of events logged
    total_events: u64,
    /// Events by type (for statistics)
    events_by_type: HashMap<AuditEventType, u64>,
}

impl AuditTrail {
    /// Create a new audit trail
    pub fn new(max_records: usize) -> Self {
        Self {
            records: VecDeque::new(),
            max_records,
            total_events: 0,
            events_by_type: HashMap::new(),
        }
    }

    /// Add a record to the audit trail
    pub fn add_record(&mut self, record: AuditRecord) {
        // Update statistics
        for event in &record.events {
            self.total_events += 1;
            *self
                .events_by_type
                .entry(event.event_type.clone())
                .or_insert(0) += 1;
        }

        // Add record
        self.records.push_back(record);

        // Maintain maximum size
        while self.records.len() > self.max_records {
            if let Some(removed) = self.records.pop_front() {
                // Update statistics when removing old records
                for event in &removed.events {
                    if let Some(count) = self.events_by_type.get_mut(&event.event_type) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            self.events_by_type.remove(&event.event_type);
                        }
                    }
                }
            }
        }
    }

    /// Get records within a time range
    pub fn get_records_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&AuditRecord> {
        self.records
            .iter()
            .filter(|record| record.start_time >= start && record.start_time <= end)
            .collect()
    }

    /// Get records by user
    pub fn get_records_by_user(&self, user_id: &str) -> Vec<&AuditRecord> {
        self.records
            .iter()
            .filter(|record| {
                record
                    .events
                    .iter()
                    .any(|event| event.user_id.as_ref().map_or(false, |id| id == user_id))
            })
            .collect()
    }

    /// Get records by event type
    pub fn get_records_by_event_type(&self, event_type: &AuditEventType) -> Vec<&AuditRecord> {
        self.records
            .iter()
            .filter(|record| {
                record
                    .events
                    .iter()
                    .any(|event| &event.event_type == event_type)
            })
            .collect()
    }

    /// Get total number of events
    pub fn total_events(&self) -> u64 {
        self.total_events
    }

    /// Get event statistics by type
    pub fn event_statistics(&self) -> &HashMap<AuditEventType, u64> {
        &self.events_by_type
    }

    /// Get recent events
    pub fn recent_events(&self, limit: usize) -> Vec<&AuditEvent> {
        self.records
            .iter()
            .rev()
            .flat_map(|record| record.events.iter())
            .take(limit)
            .collect()
    }
}

/// Node in the explanation lineage graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    /// Unique node ID
    pub id: String,
    /// Node type (data, model, explanation, etc.)
    pub node_type: String,
    /// Human-readable name
    pub name: String,
    /// When this node was created
    pub created_at: DateTime<Utc>,
    /// User who created this node
    pub created_by: Option<String>,
    /// Additional metadata about the node
    pub metadata: HashMap<String, String>,
    /// Hash or checksum for integrity verification
    pub checksum: Option<String>,
}

impl LineageNode {
    /// Create a new lineage node
    pub fn new(node_type: String, name: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            node_type,
            name,
            created_at: Utc::now(),
            created_by: None,
            metadata: HashMap::new(),
            checksum: None,
        }
    }

    /// Set creator information
    pub fn with_creator(mut self, user_id: String) -> Self {
        self.created_by = Some(user_id);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set checksum
    pub fn with_checksum(mut self, checksum: String) -> Self {
        self.checksum = Some(checksum);
        self
    }
}

/// Relationship between lineage nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageRelation {
    /// Unique relation ID
    pub id: String,
    /// Source node ID
    pub from_node: String,
    /// Target node ID
    pub to_node: String,
    /// Type of relationship
    pub relation_type: String,
    /// Operation that created this relationship
    pub operation: OperationType,
    /// When the relationship was created
    pub created_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl LineageRelation {
    /// Create a new lineage relation
    pub fn new(
        from_node: String,
        to_node: String,
        relation_type: String,
        operation: OperationType,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            from_node,
            to_node,
            relation_type,
            operation,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Complete explanation lineage tracking system
#[derive(Debug)]
pub struct ExplanationLineage {
    /// All nodes in the lineage graph
    nodes: HashMap<String, LineageNode>,
    /// All relationships between nodes
    relations: Vec<LineageRelation>,
    /// Index of relations by source node
    relations_from: HashMap<String, Vec<String>>,
    /// Index of relations by target node
    relations_to: HashMap<String, Vec<String>>,
}

impl ExplanationLineage {
    /// Create a new lineage tracker
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            relations: Vec::new(),
            relations_from: HashMap::new(),
            relations_to: HashMap::new(),
        }
    }

    /// Add a node to the lineage
    pub fn add_node(&mut self, node: LineageNode) -> SklResult<()> {
        if self.nodes.contains_key(&node.id) {
            return Err(SklearsError::InvalidParameter {
                name: "node_id".to_string(),
                reason: format!("Node '{}' already exists", node.id),
            });
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Option<&LineageNode> {
        self.nodes.get(id)
    }

    /// Add a relationship between nodes
    pub fn add_relation(&mut self, relation: LineageRelation) -> SklResult<()> {
        // Verify that both nodes exist
        if !self.nodes.contains_key(&relation.from_node) {
            return Err(SklearsError::InvalidParameter {
                name: "from_node".to_string(),
                reason: format!("Source node '{}' does not exist", relation.from_node),
            });
        }
        if !self.nodes.contains_key(&relation.to_node) {
            return Err(SklearsError::InvalidParameter {
                name: "to_node".to_string(),
                reason: format!("Target node '{}' does not exist", relation.to_node),
            });
        }

        // Add to indices
        self.relations_from
            .entry(relation.from_node.clone())
            .or_insert_with(Vec::new)
            .push(relation.id.clone());

        self.relations_to
            .entry(relation.to_node.clone())
            .or_insert_with(Vec::new)
            .push(relation.id.clone());

        self.relations.push(relation);
        Ok(())
    }

    /// Get all ancestors of a node
    pub fn get_ancestors(&self, node_id: &str) -> Vec<&LineageNode> {
        let mut ancestors = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.get_ancestors_recursive(node_id, &mut ancestors, &mut visited);
        ancestors
    }

    fn get_ancestors_recursive<'a>(
        &'a self,
        node_id: &str,
        ancestors: &mut Vec<&'a LineageNode>,
        visited: &mut std::collections::HashSet<String>,
    ) {
        if visited.contains(node_id) {
            return; // Prevent infinite loops in case of cycles
        }
        visited.insert(node_id.to_string());

        if let Some(relation_ids) = self.relations_to.get(node_id) {
            for relation_id in relation_ids {
                if let Some(relation) = self.relations.iter().find(|r| &r.id == relation_id) {
                    if let Some(parent_node) = self.nodes.get(&relation.from_node) {
                        ancestors.push(parent_node);
                        self.get_ancestors_recursive(&relation.from_node, ancestors, visited);
                    }
                }
            }
        }
    }

    /// Get all descendants of a node
    pub fn get_descendants(&self, node_id: &str) -> Vec<&LineageNode> {
        let mut descendants = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.get_descendants_recursive(node_id, &mut descendants, &mut visited);
        descendants
    }

    fn get_descendants_recursive<'a>(
        &'a self,
        node_id: &str,
        descendants: &mut Vec<&'a LineageNode>,
        visited: &mut std::collections::HashSet<String>,
    ) {
        if visited.contains(node_id) {
            return; // Prevent infinite loops
        }
        visited.insert(node_id.to_string());

        if let Some(relation_ids) = self.relations_from.get(node_id) {
            for relation_id in relation_ids {
                if let Some(relation) = self.relations.iter().find(|r| &r.id == relation_id) {
                    if let Some(child_node) = self.nodes.get(&relation.to_node) {
                        descendants.push(child_node);
                        self.get_descendants_recursive(&relation.to_node, descendants, visited);
                    }
                }
            }
        }
    }

    /// Get the complete lineage path from a source to a target
    pub fn get_lineage_path(&self, from_node: &str, to_node: &str) -> Option<Vec<&LineageNode>> {
        let mut path = Vec::new();
        let mut visited = std::collections::HashSet::new();

        if self.find_path_recursive(from_node, to_node, &mut path, &mut visited) {
            Some(path)
        } else {
            None
        }
    }

    fn find_path_recursive<'a>(
        &'a self,
        current: &str,
        target: &str,
        path: &mut Vec<&'a LineageNode>,
        visited: &mut std::collections::HashSet<String>,
    ) -> bool {
        if visited.contains(current) {
            return false;
        }
        visited.insert(current.to_string());

        if let Some(node) = self.nodes.get(current) {
            path.push(node);
        }

        if current == target {
            return true;
        }

        if let Some(relation_ids) = self.relations_from.get(current) {
            for relation_id in relation_ids {
                if let Some(relation) = self.relations.iter().find(|r| &r.id == relation_id) {
                    if self.find_path_recursive(&relation.to_node, target, path, visited) {
                        return true;
                    }
                }
            }
        }

        path.pop();
        visited.remove(current);
        false
    }

    /// Get nodes by type
    pub fn get_nodes_by_type(&self, node_type: &str) -> Vec<&LineageNode> {
        self.nodes
            .values()
            .filter(|node| node.node_type == node_type)
            .collect()
    }

    /// Get relationships by operation type
    pub fn get_relations_by_operation(&self, operation: &OperationType) -> Vec<&LineageRelation> {
        self.relations
            .iter()
            .filter(|relation| &relation.operation == operation)
            .collect()
    }

    /// Get total number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get total number of relationships
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }
}

impl Default for ExplanationLineage {
    fn default() -> Self {
        Self::new()
    }
}

/// Lineage tracker combining audit logging with lineage tracking
#[derive(Debug)]
pub struct LineageTracker {
    /// Audit trail
    audit_trail: AuditTrail,
    /// Explanation lineage
    lineage: ExplanationLineage,
}

impl LineageTracker {
    /// Create a new lineage tracker
    pub fn new(max_audit_records: usize) -> Self {
        Self {
            audit_trail: AuditTrail::new(max_audit_records),
            lineage: ExplanationLineage::new(),
        }
    }

    /// Track an operation with both audit logging and lineage
    pub fn track_operation(
        &mut self,
        operation: OperationType,
        source_nodes: Vec<String>,
        target_node: LineageNode,
        audit_event: AuditEvent,
    ) -> SklResult<()> {
        // Validate that all source nodes exist before making any changes
        for source_id in &source_nodes {
            if !self.lineage.nodes.contains_key(source_id) {
                return Err(SklearsError::InvalidParameter {
                    name: "source_node".to_string(),
                    reason: format!("Source node '{}' does not exist", source_id),
                });
            }
        }

        // Add target node to lineage
        let target_id = target_node.id.clone();
        self.lineage.add_node(target_node)?;

        // Add relationships from source nodes to target
        for source_id in source_nodes {
            let relation = LineageRelation::new(
                source_id,
                target_id.clone(),
                operation.to_string(),
                operation.clone(),
            );
            self.lineage.add_relation(relation)?;
        }

        // Add audit event
        let mut record = AuditRecord::new(Uuid::new_v4().to_string());
        record.add_event(audit_event);
        record.complete();
        self.audit_trail.add_record(record);

        Ok(())
    }

    /// Get audit trail
    pub fn audit_trail(&self) -> &AuditTrail {
        &self.audit_trail
    }

    /// Get lineage
    pub fn lineage(&self) -> &ExplanationLineage {
        &self.lineage
    }
}

/// Audit logger for recording events
#[derive(Debug)]
pub struct AuditLogger {
    /// Current audit trail
    trail: AuditTrail,
    /// Configuration
    config: AuditLoggerConfig,
}

/// Configuration for audit logging
#[derive(Debug, Clone)]
pub struct AuditLoggerConfig {
    /// Maximum number of records to keep in memory
    pub max_records: usize,
    /// Whether to log successful operations
    pub log_success: bool,
    /// Whether to log failed operations
    pub log_failures: bool,
    /// Minimum severity level to log
    pub min_severity: AuditSeverity,
    /// Whether to include client information
    pub include_client_info: bool,
}

impl Default for AuditLoggerConfig {
    fn default() -> Self {
        Self {
            max_records: 10000,
            log_success: true,
            log_failures: true,
            min_severity: AuditSeverity::Info,
            include_client_info: true,
        }
    }
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(config: AuditLoggerConfig) -> Self {
        Self {
            trail: AuditTrail::new(config.max_records),
            config,
        }
    }

    /// Log an event
    pub fn log_event(&mut self, event: AuditEvent) {
        // Filter by severity
        if !self.should_log(&event) {
            return;
        }

        let mut record = AuditRecord::new(event.id.clone());
        record.add_event(event);
        record.complete();
        self.trail.add_record(record);
    }

    /// Log multiple related events as a transaction
    pub fn log_transaction(&mut self, transaction_id: String, events: Vec<AuditEvent>) {
        if events.is_empty() {
            return;
        }

        // Filter events by configuration
        let filtered_events: Vec<AuditEvent> = events
            .into_iter()
            .filter(|event| self.should_log(event))
            .collect();

        if filtered_events.is_empty() {
            return;
        }

        let mut record = AuditRecord::new(transaction_id);
        for event in filtered_events {
            record.add_event(event);
        }
        record.complete();
        self.trail.add_record(record);
    }

    fn should_log(&self, event: &AuditEvent) -> bool {
        // Check success/failure logging settings
        if !self.config.log_success && event.success {
            return false;
        }
        if !self.config.log_failures && !event.success {
            return false;
        }

        // Check severity level
        let severity_level = match event.severity {
            AuditSeverity::Info => 0,
            AuditSeverity::Warning => 1,
            AuditSeverity::Error => 2,
            AuditSeverity::Critical => 3,
        };

        let min_severity_level = match self.config.min_severity {
            AuditSeverity::Info => 0,
            AuditSeverity::Warning => 1,
            AuditSeverity::Error => 2,
            AuditSeverity::Critical => 3,
        };

        severity_level >= min_severity_level
    }

    /// Get the audit trail
    pub fn trail(&self) -> &AuditTrail {
        &self.trail
    }
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::DataIngestion => write!(f, "data_ingestion"),
            OperationType::DataTransformation => write!(f, "data_transformation"),
            OperationType::ModelTraining => write!(f, "model_training"),
            OperationType::ModelInference => write!(f, "model_inference"),
            OperationType::ExplanationGeneration => write!(f, "explanation_generation"),
            OperationType::ExplanationTransformation => write!(f, "explanation_transformation"),
            OperationType::ExplanationAggregation => write!(f, "explanation_aggregation"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_creation() {
        let event = AuditEvent::new(
            AuditEventType::ExplanationGenerated,
            "generate_explanation".to_string(),
            "Generated SHAP explanation".to_string(),
        );

        assert_eq!(event.event_type, AuditEventType::ExplanationGenerated);
        assert_eq!(event.operation, "generate_explanation");
        assert!(event.success);
        assert!(event.error_message.is_none());
    }

    #[test]
    fn test_audit_event_severity() {
        assert_eq!(
            AuditEventType::ExplanationGenerated.severity(),
            AuditSeverity::Info
        );
        assert_eq!(
            AuditEventType::AccessDenied.severity(),
            AuditSeverity::Warning
        );
        assert_eq!(AuditEventType::SystemError.severity(), AuditSeverity::Error);
        assert_eq!(
            AuditEventType::SystemStartup.severity(),
            AuditSeverity::Critical
        );
    }

    #[test]
    fn test_audit_record() {
        let mut record = AuditRecord::new("transaction_1".to_string());
        assert!(record.success);

        let event1 = AuditEvent::new(
            AuditEventType::ExplanationGenerated,
            "generate".to_string(),
            "Generated explanation".to_string(),
        );
        record.add_event(event1);

        let event2 = AuditEvent::new(
            AuditEventType::SystemError,
            "error".to_string(),
            "An error occurred".to_string(),
        )
        .with_error("Test error".to_string());

        record.add_event(event2);
        assert!(!record.success); // Should be false due to error event
    }

    #[test]
    fn test_audit_trail() {
        let mut trail = AuditTrail::new(2);

        let mut record1 = AuditRecord::new("tx1".to_string());
        record1.add_event(AuditEvent::new(
            AuditEventType::ExplanationGenerated,
            "generate".to_string(),
            "Test".to_string(),
        ));
        trail.add_record(record1);

        assert_eq!(trail.total_events(), 1);
        assert_eq!(
            trail
                .event_statistics()
                .get(&AuditEventType::ExplanationGenerated),
            Some(&1)
        );
    }

    #[test]
    fn test_lineage_node() {
        let node = LineageNode::new("explanation".to_string(), "SHAP explanation".to_string())
            .with_creator("user1".to_string())
            .with_metadata("model_type".to_string(), "random_forest".to_string());

        assert_eq!(node.node_type, "explanation");
        assert_eq!(node.name, "SHAP explanation");
        assert_eq!(node.created_by, Some("user1".to_string()));
        assert_eq!(
            node.metadata.get("model_type"),
            Some(&"random_forest".to_string())
        );
    }

    #[test]
    fn test_lineage_relation() {
        let relation = LineageRelation::new(
            "model1".to_string(),
            "explanation1".to_string(),
            "generates".to_string(),
            OperationType::ExplanationGeneration,
        )
        .with_metadata("method".to_string(), "shap".to_string());

        assert_eq!(relation.from_node, "model1");
        assert_eq!(relation.to_node, "explanation1");
        assert_eq!(relation.relation_type, "generates");
        assert_eq!(relation.operation, OperationType::ExplanationGeneration);
    }

    #[test]
    fn test_explanation_lineage() {
        let mut lineage = ExplanationLineage::new();

        let data_node = LineageNode::new("data".to_string(), "Training data".to_string());
        let model_node = LineageNode::new("model".to_string(), "RF Model".to_string());
        let explanation_node =
            LineageNode::new("explanation".to_string(), "SHAP values".to_string());

        let data_id = data_node.id.clone();
        let model_id = model_node.id.clone();
        let explanation_id = explanation_node.id.clone();

        lineage.add_node(data_node).unwrap();
        lineage.add_node(model_node).unwrap();
        lineage.add_node(explanation_node).unwrap();

        // Data -> Model
        let relation1 = LineageRelation::new(
            data_id.clone(),
            model_id.clone(),
            "trains".to_string(),
            OperationType::ModelTraining,
        );
        lineage.add_relation(relation1).unwrap();

        // Model -> Explanation
        let relation2 = LineageRelation::new(
            model_id.clone(),
            explanation_id.clone(),
            "generates".to_string(),
            OperationType::ExplanationGeneration,
        );
        lineage.add_relation(relation2).unwrap();

        // Test ancestor/descendant relationships
        let ancestors = lineage.get_ancestors(&explanation_id);
        assert_eq!(ancestors.len(), 2); // Model and data

        let descendants = lineage.get_descendants(&data_id);
        assert_eq!(descendants.len(), 2); // Model and explanation

        // Test lineage path
        let path = lineage.get_lineage_path(&data_id, &explanation_id).unwrap();
        assert_eq!(path.len(), 3); // Data -> Model -> Explanation
    }

    #[test]
    fn test_audit_logger() {
        let config = AuditLoggerConfig::default();
        let mut logger = AuditLogger::new(config);

        let event = AuditEvent::new(
            AuditEventType::ExplanationGenerated,
            "generate".to_string(),
            "Test explanation".to_string(),
        );

        logger.log_event(event);
        assert_eq!(logger.trail().total_events(), 1);
    }

    #[test]
    fn test_lineage_tracker() {
        let mut tracker = LineageTracker::new(100);

        let target_node =
            LineageNode::new("explanation".to_string(), "Test explanation".to_string());
        let event = AuditEvent::new(
            AuditEventType::ExplanationGenerated,
            "generate".to_string(),
            "Generated explanation".to_string(),
        );

        tracker
            .track_operation(
                OperationType::ExplanationGeneration,
                vec!["model1".to_string()],
                target_node,
                event,
            )
            .unwrap_err(); // Should fail because model1 doesn't exist

        // Add model node first
        let model_node = LineageNode::new("model".to_string(), "Test model".to_string());
        let model_id = model_node.id.clone();
        tracker.lineage.add_node(model_node).unwrap();

        let target_node =
            LineageNode::new("explanation".to_string(), "Test explanation".to_string());
        let event = AuditEvent::new(
            AuditEventType::ExplanationGenerated,
            "generate".to_string(),
            "Generated explanation".to_string(),
        );

        tracker
            .track_operation(
                OperationType::ExplanationGeneration,
                vec![model_id],
                target_node,
                event,
            )
            .unwrap();

        assert_eq!(tracker.audit_trail().total_events(), 1);
        assert_eq!(tracker.lineage().node_count(), 2);
        assert_eq!(tracker.lineage().relation_count(), 1);
    }
}
